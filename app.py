from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List
import shutil
import subprocess
import os
import sys
import uuid
import threading
import time
import sqlite3
import jwt
import bcrypt
import stripe
import logging
from contextlib import contextmanager
import queue
from threading import Semaphore, Lock
import hashlib
import io
import json

# PDF预览图生成依赖
try:
    import fitz  # PyMuPDF
    from PIL import Image, ImageDraw, ImageFont
    PDF_PREVIEW_AVAILABLE = True
except ImportError:
    PDF_PREVIEW_AVAILABLE = False
    # 注意：logger 还未定义，使用 print 或延迟警告

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
BUILD_ROOT = BASE_DIR / "build_jobs"
BUILD_ROOT.mkdir(exist_ok=True)
BUILD_DIR = BASE_DIR / "build"
UPLOAD_SESSIONS_DIR = BASE_DIR / "upload_sessions"  # 文件上传暂存目录
UPLOAD_SESSIONS_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "data.db"

JWT_SECRET = os.environ.get("JWT_SECRET", "change_me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_HOURS = 24
security = HTTPBearer(auto_error=False)
# Token黑名单改用数据库表（见init_db中的revoked_tokens表）

# Stripe 配置
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", STRIPE_API_KEY)  # 兼容旧的环境变量名
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")  # $0.99 下载解锁的价格ID
stripe.api_key = STRIPE_SECRET_KEY if STRIPE_SECRET_KEY else None

PRICE_IDS = {
    "starter": os.environ.get("STRIPE_PRICE_STARTER", ""),
    "pro": os.environ.get("STRIPE_PRICE_PRO", ""),
}

# Mock 支付配置（仅用于开发/测试环境）
ENABLE_MOCK_PAYMENT = os.environ.get("ENABLE_MOCK_PAYMENT", "false").lower() == "true"
# 生产环境检查：如果域名是生产域名，强制禁用 mock
PRODUCTION_DOMAINS = ["examfrompdf.com", "www.examfrompdf.com"]
IS_PRODUCTION = any(domain in os.environ.get("API_BASE_URL", "") for domain in PRODUCTION_DOMAINS)
if IS_PRODUCTION and ENABLE_MOCK_PAYMENT:
    logger.warning("Mock payment is disabled in production environment")
    ENABLE_MOCK_PAYMENT = False

PLAN_LIMITS = {
    "free": {"daily": 9999, "monthly": 9999},  # 测试阶段：临时提高限额
    "starter": {"daily": 999999, "monthly": 10},
    "pro": {"daily": 999999, "monthly": 50},
}

ALLOWED_ORIGINS = [
    "https://examfrompdf.com",
    "https://www.examfrompdf.com",
    "https://examfrompdfcom.lovable.app",  # 已发布域名
    "https://672fee75-aabe-4860-9b0b-5f907b22109b.lovableproject.com",  # 预览域名
]
ALLOWED_ORIGIN_REGEX = r"https://.*\.lovable\.app"

app = FastAPI(title="ExamFromPDF")

# 添加请求验证错误处理器，提供更详细的错误信息
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """捕获422验证错误，记录详细信息用于调试"""
    try:
        body = await request.body()
        body_str = body.decode('utf-8', errors='ignore')[:500]  # 限制长度
    except:
        body_str = "Unable to read body"
    
    logger.error(f"[VALIDATION ERROR] Path: {request.url.path}, Method: {request.method}")
    logger.error(f"[VALIDATION ERROR] Content-Type: {request.headers.get('content-type', 'N/A')}")
    logger.error(f"[VALIDATION ERROR] Errors: {exc.errors()}")
    logger.error(f"[VALIDATION ERROR] Body preview: {body_str}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Request validation failed. Please check that you're sending multipart/form-data with 'file' and 'session_id' fields."
        }
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 移除内存状态，完全依赖数据库（商用级改进）
# 所有任务状态从数据库读取，支持多实例部署和持久化


# -------------------- 数据库工具（商用级改进） --------------------
# 使用上下文管理器确保连接正确关闭
@contextmanager
def get_db():
    """数据库连接上下文管理器，确保连接正确关闭，并监控操作耗时"""
    start_time = time.time()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
        elapsed = time.time() - start_time
        # 如果操作耗时超过100ms，记录警告
        if elapsed > 0.1:
            logger.warning(f"Database operation took {elapsed:.3f}s (slow)")
    except Exception as e:
        conn.rollback()
        elapsed = time.time() - start_time
        logger.error(f"Database error after {elapsed:.3f}s: {e}", exc_info=True)
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plan TEXT NOT NULL,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT UNIQUE,
                status TEXT,
                current_period_end INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(user_id, date),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                user_id INTEGER,  -- 允许NULL，用于匿名用户
                device_fingerprint TEXT,  -- 匿名用户的设备指纹
                file_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                download_url TEXT,
                error TEXT,
                updated_at TEXT,
                mcq_count INTEGER DEFAULT 10,  -- 选择题数量
                short_answer_count INTEGER DEFAULT 3,  -- 简答题数量
                long_question_count INTEGER DEFAULT 1,  -- 论述题数量
                difficulty TEXT DEFAULT 'medium',  -- 难度等级: easy, medium, hard
                special_requests TEXT,  -- 用户特殊要求
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                revoked_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        # 匿名用户追踪表（基于 anon_id + 日期）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS anon_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anon_id TEXT NOT NULL,
                date TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(anon_id, date)
            );
            """
        )
        # 匿名用户到注册用户的关联表（用于注册时关联使用记录）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS anon_to_user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anon_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                associated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                UNIQUE(anon_id, user_id)
            );
            """
        )
        # 注册奖励表（记录用户获得的注册奖励次数）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS registration_bonus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                bonus_count INTEGER NOT NULL DEFAULT 0,
                used_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                UNIQUE(user_id)
            );
            """
        )
        # 创建索引提升查询性能
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_anon_usage_anon_id ON anon_usage(anon_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_anon_usage_date ON anon_usage(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_anon_to_user_anon_id ON anon_to_user(anon_id)")


init_db()


# -------------------- 数据库迁移 --------------------
def migrate_db():
    """数据库迁移：添加缺失的列"""
    with get_db() as conn:
        cur = conn.cursor()
        
        # 检查 jobs 表的列
        cur.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cur.fetchall()]
        
        # 添加 updated_at 列（如果不存在）
        if "updated_at" not in columns:
            logger.info("Adding updated_at column to jobs table")
            try:
                cur.execute("ALTER TABLE jobs ADD COLUMN updated_at TEXT")
                logger.info("Successfully added updated_at column")
            except Exception as e:
                logger.error(f"Failed to add updated_at column: {e}", exc_info=True)
        
        # 添加 device_fingerprint 列（如果不存在）- 用于匿名用户
        if "device_fingerprint" not in columns:
            logger.info("Adding device_fingerprint column to jobs table")
            try:
                cur.execute("ALTER TABLE jobs ADD COLUMN device_fingerprint TEXT")
                logger.info("Successfully added device_fingerprint column")
            except Exception as e:
                logger.error(f"Failed to add device_fingerprint column: {e}", exc_info=True)
        
        # 重新获取列列表（因为可能已经添加了新列）
        cur.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cur.fetchall()]
        
        # 添加exam配置列（如果不存在）
        for col_name, col_type, default in [
            ("mcq_count", "INTEGER", 10),
            ("short_answer_count", "INTEGER", 3),
            ("long_question_count", "INTEGER", 1),
            ("difficulty", "TEXT", "medium")
        ]:
            if col_name not in columns:
                logger.info(f"Adding {col_name} column to jobs table")
                try:
                    if col_type == "INTEGER":
                        cur.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} INTEGER DEFAULT {default}")
                    else:
                        cur.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} TEXT DEFAULT '{default}'")
                    logger.info(f"Successfully added {col_name} column")
                except Exception as e:
                    logger.error(f"Failed to add {col_name} column: {e}", exc_info=True)
        
        # 重新获取列列表（因为可能已经添加了新列）
        cur.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cur.fetchall()]
        
        # 添加 special_requests 列（如果不存在）
        if "special_requests" not in columns:
            logger.info("Adding special_requests column to jobs table")
            try:
                cur.execute("ALTER TABLE jobs ADD COLUMN special_requests TEXT")
                logger.info("Successfully added special_requests column")
            except Exception as e:
                logger.error(f"Failed to add special_requests column: {e}", exc_info=True)
        
        # 重新获取列列表（因为可能已经添加了新列）
        cur.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cur.fetchall()]
        
        # 添加监控相关的时间戳字段（如果不存在）
        for col_name in ["started_at", "completed_at"]:
            if col_name not in columns:
                logger.info(f"Adding {col_name} column to jobs table for monitoring")
                try:
                    cur.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} TEXT")
                    logger.info(f"Successfully added {col_name} column")
                except Exception as e:
                    logger.error(f"Failed to add {col_name} column: {e}", exc_info=True)
        
        # 重新获取列列表（因为可能已经添加了新列）
        cur.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cur.fetchall()]
        
        # 添加进度信息字段（如果不存在）
        if "progress_info" not in columns:
            logger.info("Adding progress_info column to jobs table for progress tracking")
            try:
                cur.execute("ALTER TABLE jobs ADD COLUMN progress_info TEXT")
                logger.info("Successfully added progress_info column")
            except Exception as e:
                logger.error(f"Failed to add progress_info column: {e}", exc_info=True)
        
        # 添加 answer_status 列（如果不存在）- 用于跟踪答案生成状态
        if "answer_status" not in columns:
            logger.info("Adding answer_status column to jobs table")
            try:
                cur.execute("ALTER TABLE jobs ADD COLUMN answer_status TEXT DEFAULT 'pending'")
                logger.info("Successfully added answer_status column")
            except Exception as e:
                logger.error(f"Failed to add answer_status column: {e}", exc_info=True)
        
        # 重新获取列列表（因为可能已经添加了新列）
        cur.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cur.fetchall()]
        
        # 添加支付解锁相关字段（如果不存在）
        if "is_unlocked" not in columns:
            logger.info("Adding is_unlocked column to jobs table")
            try:
                cur.execute("ALTER TABLE jobs ADD COLUMN is_unlocked INTEGER DEFAULT 0")  # SQLite uses INTEGER for boolean
                logger.info("Successfully added is_unlocked column")
            except Exception as e:
                logger.error(f"Failed to add is_unlocked column: {e}", exc_info=True)
        
        if "unlocked_at" not in columns:
            logger.info("Adding unlocked_at column to jobs table")
            try:
                cur.execute("ALTER TABLE jobs ADD COLUMN unlocked_at TEXT")
                logger.info("Successfully added unlocked_at column")
            except Exception as e:
                logger.error(f"Failed to add unlocked_at column: {e}", exc_info=True)
        
        # 检查并创建 transactions 表（如果不存在）
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
        if not cur.fetchone():
            logger.info("Creating transactions table")
            try:
                cur.execute(
                    """
                    CREATE TABLE transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        user_id INTEGER,
                        stripe_session_id TEXT UNIQUE NOT NULL,
                        amount INTEGER NOT NULL,
                        currency TEXT NOT NULL DEFAULT 'usd',
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        completed_at TEXT,
                        FOREIGN KEY(job_id) REFERENCES jobs(id),
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    );
                    """
                )
                # 创建索引
                cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_job_id ON transactions(job_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_session_id ON transactions(stripe_session_id)")
                logger.info("Successfully created transactions table")
            except Exception as e:
                logger.error(f"Failed to create transactions table: {e}", exc_info=True)
        
        # 检查 user_id 列是否允许 NULL（SQLite不支持直接修改NOT NULL约束，需要重建表）
        cur.execute("PRAGMA table_info(jobs)")
        columns_info = cur.fetchall()
        user_id_not_null = False
        for col in columns_info:
            if col[1] == "user_id":
                user_id_not_null = col[3] == 1  # 1 means NOT NULL, 0 means nullable
                break
        
        # 如果 user_id 是 NOT NULL，需要重建表以允许 NULL（用于匿名用户）
        if user_id_not_null:
            logger.info("Rebuilding jobs table to allow NULL user_id for anonymous users")
            try:
                # 1. 创建新表（允许 user_id 为 NULL，包含exam配置列）
                cur.execute("""
                    CREATE TABLE jobs_new (
                        id TEXT PRIMARY KEY,
                        user_id INTEGER,
                        device_fingerprint TEXT,
                        file_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        download_url TEXT,
                        error TEXT,
                        updated_at TEXT,
                        mcq_count INTEGER DEFAULT 10,
                        short_answer_count INTEGER DEFAULT 3,
                        long_question_count INTEGER DEFAULT 1,
                        difficulty TEXT DEFAULT 'medium',
                        special_requests TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )
                """)
                
                # 2. 复制数据（保留所有现有数据，新列使用默认值）
                cur.execute("""
                    INSERT INTO jobs_new 
                    (id, user_id, device_fingerprint, file_name, status, created_at, download_url, error, updated_at, mcq_count, short_answer_count, long_question_count, difficulty)
                    SELECT 
                        id, 
                        user_id,
                        device_fingerprint,
                        file_name, 
                        status, 
                        created_at, 
                        download_url, 
                        error,
                        COALESCE(updated_at, created_at),
                        COALESCE(mcq_count, 10),
                        COALESCE(short_answer_count, 3),
                        COALESCE(long_question_count, 1),
                        COALESCE(difficulty, 'medium')
                    FROM jobs
                """)
                
                # 3. 删除旧表
                cur.execute("DROP TABLE jobs")
                
                # 4. 重命名新表
                cur.execute("ALTER TABLE jobs_new RENAME TO jobs")
                
                # 5. 重新创建索引
                cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")
                
                logger.info("Successfully rebuilt jobs table with nullable user_id")
            except Exception as e:
                logger.error(f"Failed to rebuild jobs table: {e}", exc_info=True)
                conn.rollback()
                raise
        
        conn.commit()


# 运行迁移
migrate_db()


# -------------------- 任务队列和并发控制（阶段1改进） --------------------
# 并发限制：最多同时处理的任务数
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))  # 默认5个并发
job_semaphore = Semaphore(MAX_CONCURRENT_JOBS)

# 任务队列：排队等待执行的任务
job_queue = queue.Queue()

# 队列监控
queue_stats = {
    "current_queue_size": 0,
    "max_queue_size": 0,
    "current_processing": 0,
    "max_processing": 0,
    "total_processed": 0,
    "total_failed": 0,
}
queue_stats_lock = Lock()  # 保护队列统计数据的锁

def update_queue_stats(action: str, value: int = 1):
    """更新队列统计信息"""
    with queue_stats_lock:
        if action == "enqueue":
            queue_stats["current_queue_size"] += value
            queue_stats["max_queue_size"] = max(
                queue_stats["max_queue_size"], 
                queue_stats["current_queue_size"]
            )
        elif action == "dequeue":
            queue_stats["current_queue_size"] = max(0, queue_stats["current_queue_size"] - value)
        elif action == "start_processing":
            queue_stats["current_processing"] += value
            queue_stats["max_processing"] = max(
                queue_stats["max_processing"],
                queue_stats["current_processing"]
            )
        elif action == "finish_processing":
            queue_stats["current_processing"] = max(0, queue_stats["current_processing"] - value)
            queue_stats["total_processed"] += value
        elif action == "fail_processing":
            queue_stats["current_processing"] = max(0, queue_stats["current_processing"] - value)
            queue_stats["total_failed"] += value

def worker_thread():
    """
    工作线程：从队列中取出任务并执行
    多个任务会排队执行，但最多同时处理 MAX_CONCURRENT_JOBS 个
    """
    while True:
        try:
            # 从队列中获取任务（阻塞等待）
            item = job_queue.get()
            # 兼容新旧格式
            if len(item) == 3:
                job_id, lecture_paths, exam_config = item
                # 兼容旧格式：如果 lecture_paths 是单个 Path 对象，转换为列表
                if isinstance(lecture_paths, Path):
                    lecture_paths = [lecture_paths]
            else:
                job_id, lecture_path = item[:2]
                lecture_paths = [lecture_path] if isinstance(lecture_path, Path) else lecture_path
                exam_config = None
            
            # 获取信号量（如果已达到最大并发数，会阻塞等待）
            job_semaphore.acquire()
            update_queue_stats("dequeue")
            update_queue_stats("start_processing")
            
            try:
                logger.info(f"Worker thread: Starting job {job_id} with {len(lecture_paths)} PDF file(s)")
                run_job(job_id, lecture_paths, exam_config)
                update_queue_stats("finish_processing")
                logger.info(f"Worker thread: Completed job {job_id}")
            except Exception as e:
                update_queue_stats("fail_processing")
                logger.error(f"Worker thread: Job {job_id} failed: {e}", exc_info=True)
            finally:
                # 释放信号量，允许下一个任务开始
                job_semaphore.release()
                job_queue.task_done()
                
        except Exception as e:
            logger.error(f"Worker thread error: {e}", exc_info=True)

# 启动工作线程（守护线程，主进程退出时自动退出）
worker = threading.Thread(target=worker_thread, daemon=True)
worker.start()
logger.info(f"Task queue worker thread started (max concurrent: {MAX_CONCURRENT_JOBS})")


# -------------------- 自动清理任务（7天过期） --------------------
def cleanup_old_jobs():
    """
    清理超过7天的job记录和文件，以及超过1小时的未使用session目录
    每天运行一次
    """
    while True:
        try:
            # 等待24小时（86400秒）
            time.sleep(86400)  # 24小时
            
            logger.info("Starting cleanup of old jobs (7 days retention) and old sessions (1 hour retention)")
            cutoff_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            
            deleted_count = 0
            deleted_files_count = 0
            deleted_sessions_count = 0
            errors = []
            
            # 清理过期的session目录（超过1小时未使用）
            if UPLOAD_SESSIONS_DIR.exists():
                session_cutoff_time = time.time() - 3600  # 1小时前
                for session_dir in UPLOAD_SESSIONS_DIR.iterdir():
                    if session_dir.is_dir():
                        try:
                            # 检查目录的最后修改时间
                            mtime = session_dir.stat().st_mtime
                            if mtime < session_cutoff_time:
                                shutil.rmtree(session_dir)
                                deleted_sessions_count += 1
                                logger.debug(f"Deleted old session directory: {session_dir.name}")
                        except Exception as e:
                            error_msg = f"Failed to delete session {session_dir.name}: {str(e)}"
                            errors.append(error_msg)
                            logger.warning(error_msg)
            
            with get_db() as conn:
                cur = conn.cursor()
                # 查询所有超过7天的job
                cur.execute(
                    "SELECT id, created_at, file_name FROM jobs WHERE created_at < ?",
                    (cutoff_date,)
                )
                old_jobs = cur.fetchall()
                
                logger.info(f"Found {len(old_jobs)} job(s) older than 7 days")
                
                for job_row in old_jobs:
                    job_id = job_row["id"]
                    created_at = job_row["created_at"]
                    file_name = job_row["file_name"]
                    
                    try:
                        # 删除文件目录
                        job_dir = BUILD_ROOT / job_id
                        if job_dir.exists():
                            shutil.rmtree(job_dir)
                            deleted_files_count += 1
                            logger.debug(f"Deleted files for job {job_id} (created: {created_at})")
                        else:
                            logger.debug(f"Job directory not found for job {job_id}, skipping file deletion")
                        
                        # 删除数据库记录
                        cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                        deleted_count += 1
                        logger.debug(f"Deleted database record for job {job_id}")
                        
                    except Exception as e:
                        error_msg = f"Failed to delete job {job_id}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg, exc_info=True)
                        # 即使文件删除失败，也尝试删除数据库记录
                        try:
                            cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                            deleted_count += 1
                        except:
                            pass
                
                conn.commit()
            
            logger.info(
                f"Cleanup completed: {deleted_count} job(s) deleted from database, "
                f"{deleted_files_count} file directory(ies) deleted. "
                f"Errors: {len(errors)}"
            )
            
            if errors:
                logger.warning(f"Cleanup errors: {errors}")
                
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}", exc_info=True)
            # 即使出错也继续运行，等待下次清理

# 启动清理线程（守护线程，每天运行一次）
cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
cleanup_thread.start()
logger.info("Cleanup thread started (7 days retention policy)")


# -------------------- 工具函数 --------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def generate_jwt(user_id: int, email: str) -> str:
    jti = uuid.uuid4().hex
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRES_HOURS),
        "jti": jti,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")
        # 检查数据库中的黑名单（商用级改进）
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT jti FROM revoked_tokens WHERE jti = ?", (jti,))
            if cur.fetchone():
                raise HTTPException(status_code=401, detail="Token revoked")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
        return cur.fetchone()


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()


def upsert_usage(user_id: int):
    today = date.today().isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO usage (user_id, date, count) VALUES (?, ?, 1)
            ON CONFLICT(user_id, date) DO UPDATE SET count = count + 1
            """,
            (user_id, today),
        )


def get_usage_counts(user_id: int):
    today = date.today().isoformat()
    month_prefix = today[:7]  # YYYY-MM
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT count FROM usage WHERE user_id = ? AND date = ?", (user_id, today))
        row = cur.fetchone()
        daily_count = row["count"] if row else 0
        cur.execute(
            "SELECT SUM(count) as total FROM usage WHERE user_id = ? AND date LIKE ?",
            (user_id, f"{month_prefix}-%"),
        )
        row2 = cur.fetchone()
        monthly_count = row2["total"] if row2 and row2["total"] else 0
    return daily_count, monthly_count


def get_active_subscription(user_id: int) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM subscriptions
            WHERE user_id = ?
            ORDER BY current_period_end DESC NULLS LAST
            LIMIT 1
            """,
            (user_id,),
        )
        return cur.fetchone()


def get_plan_for_user(user_id: int) -> str:
    sub = get_active_subscription(user_id)
    if sub and sub.get("status") in ("active", "trialing", "past_due"):
        return sub.get("plan", "free")
    return "free"


def check_usage_limit(user_id: int):
    plan = get_plan_for_user(user_id)
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    daily_count, monthly_count = get_usage_counts(user_id)

    daily_limit = limits.get("daily")
    monthly_limit = limits.get("monthly")

    if daily_limit is not None and daily_count >= daily_limit:
        raise HTTPException(status_code=429, detail="Daily quota exceeded")
    if monthly_limit is not None and monthly_count >= monthly_limit:
        raise HTTPException(status_code=429, detail="Monthly quota exceeded")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    token = credentials.credentials
    payload = decode_jwt(token)
    user_id = int(payload.get("sub"))
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"id": user["id"], "email": user["email"], "token_jti": payload.get("jti")}


def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """可选认证：如果有token则返回用户，否则返回None"""
    if not credentials:
        return None
    try:
        token = credentials.credentials
        payload = decode_jwt(token)
        user_id = int(payload.get("sub"))
        user = get_user_by_id(user_id)
        if not user:
            return None
        return {"id": user["id"], "email": user["email"], "token_jti": payload.get("jti")}
    except Exception:
        return None


def get_device_fingerprint(request: Request) -> str:
    """生成设备指纹：IP + User-Agent（用于备用验证）"""
    ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    # 使用hash确保隐私，同时保持一致性
    fingerprint_str = f"{ip}:{user_agent}"
    fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]  # 32字符足够
    return fingerprint


def get_or_create_anon_id(request: Request, response) -> str:
    """获取或创建匿名用户ID（从Cookie读取或生成新的）"""
    # 从Cookie读取 anon_id
    anon_id = request.cookies.get("anon_id")
    
    if not anon_id:
        # 生成新的 anon_id
        anon_id = str(uuid.uuid4())
        # 设置Cookie（强制性，不需要用户同意）
        # response 应该是 FastAPI 的 Response 对象
        if hasattr(response, 'set_cookie'):
            response.set_cookie(
                key="anon_id",
                value=anon_id,
                httponly=True,  # 防止XSS攻击
                samesite="lax",  # 防止CSRF攻击
                max_age=31536000,  # 1年过期
                secure=False  # 如果使用HTTPS，可以设置为True
            )
        logger.info(f"Generated new anon_id: {anon_id[:8]}...")
    
    return anon_id


def check_anon_usage_limit(anon_id: str) -> bool:
    """检查匿名用户是否超过限制（每天只能使用1次）"""
    today = date.today().isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT used FROM anon_usage 
            WHERE anon_id = ? AND date = ?
            """,
            (anon_id, today)
        )
        row = cur.fetchone()
        if row:
            used = row[0]
            if used >= 1:  # 匿名用户每天只能使用1次
                return False
        return True


def record_anon_usage(anon_id: str):
    """记录匿名用户使用"""
    today = date.today().isoformat()
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO anon_usage (anon_id, date, used, created_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(anon_id, date) DO UPDATE SET
                used = 1
            """,
            (anon_id, today, now)
        )


def check_guest_usage_limit(device_fingerprint: str) -> bool:
    """检查匿名用户是否超过限制（每天只能使用1次）"""
    today = date.today().isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT count FROM guest_usage 
            WHERE device_fingerprint = ? AND date = ?
            """,
            (device_fingerprint, today)
        )
        row = cur.fetchone()
        if row:
            count = row[0]
            if count >= 1:  # 匿名用户每天只能使用1次
                return False
        return True


def record_guest_usage(device_fingerprint: str, ip_address: str, user_agent: str):
    """记录匿名用户使用"""
    today = date.today().isoformat()
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO guest_usage (device_fingerprint, ip_address, user_agent, date, count, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(device_fingerprint, date) DO UPDATE SET
                count = count + 1,
                ip_address = ?,
                user_agent = ?
            """,
            (device_fingerprint, ip_address, user_agent, today, now, ip_address, user_agent)
        )


def associate_anon_usage_to_user(anon_id: str, user_id: int) -> Dict[str, Any]:
    """用户注册时，关联匿名使用记录到用户账户，并给注册奖励"""
    today = date.today().isoformat()
    now = datetime.utcnow().isoformat()
    
    with get_db() as conn:
        cur = conn.cursor()
        # 检查今天是否有匿名使用记录
        cur.execute(
            """
            SELECT used FROM anon_usage 
            WHERE anon_id = ? AND date = ?
            """,
            (anon_id, today)
        )
        row = cur.fetchone()
        
        has_used_today = row and row[0] >= 1
        
        # 关联匿名使用记录
        cur.execute(
            """
            INSERT OR IGNORE INTO anon_to_user (anon_id, user_id, associated_at)
            VALUES (?, ?, ?)
            """,
            (anon_id, user_id, now)
        )
        
        # 将匿名用户创建的job的user_id更新为当前用户
        cur.execute(
            """
            UPDATE jobs 
            SET user_id = ? 
            WHERE device_fingerprint IN (
                SELECT device_fingerprint FROM jobs 
                WHERE user_id IS NULL 
                LIMIT 100
            ) AND user_id IS NULL
            """,
            (user_id,)
        )
        updated_jobs = cur.rowcount
        
        # 如果今天已使用，记录到用户使用记录（但不给额外额度）
        if has_used_today:
            cur.execute(
                """
                INSERT INTO usage (user_id, date, count)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id, date) DO UPDATE SET
                    count = count + 1
                """,
                (user_id, today)
            )
        
        # 给注册奖励：3次额外试卷（不限当天）
        bonus_count = 3
        cur.execute(
            """
            INSERT INTO registration_bonus (user_id, bonus_count, used_count, created_at)
            VALUES (?, ?, 0, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                bonus_count = bonus_count + ?
            """,
            (user_id, bonus_count, now, bonus_count)
        )
        
        logger.info(f"Associated anon_id {anon_id[:8]}... to user {user_id}, updated {updated_jobs} jobs, gave {bonus_count} bonus")
        
        return {
            "has_used_today": has_used_today,
            "updated_jobs": updated_jobs,
            "bonus_count": bonus_count
        }
    
    return {"has_used_today": False, "updated_jobs": 0, "bonus_count": 0}


def get_registration_bonus_remaining(user_id: int) -> int:
    """获取用户剩余的注册奖励次数"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT bonus_count, used_count FROM registration_bonus 
            WHERE user_id = ?
            """,
            (user_id,)
        )
        row = cur.fetchone()
        if row:
            return max(0, row[0] - row[1])
        return 0


def use_registration_bonus(user_id: int) -> bool:
    """使用一次注册奖励"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE registration_bonus 
            SET used_count = used_count + 1
            WHERE user_id = ? AND used_count < bonus_count
            """,
            (user_id,)
        )
        return cur.rowcount > 0


# -------------------- 认证 API --------------------
@app.post("/auth/register")
async def register(request: Request, payload: Dict[str, str]):
    from fastapi import Response
    response = Response()
    
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    if get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Email already registered")

    password_hash = hash_password(password)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, password_hash, datetime.utcnow().isoformat()),
        )
        user_id = cur.lastrowid
    
    # 简化注册逻辑：只创建用户账户，不关联匿名jobs，不给注册奖励
    logger.info(f"User registered: {email} (id: {user_id})")

    token = generate_jwt(user_id, email)
    plan = get_plan_for_user(user_id)
    
    result = {
        "access_token": token,
        "user": {
            "id": user_id,
            "email": email,
            "plan": plan
        }
    }
    
    # 返回包含Cookie的响应
    from fastapi.responses import JSONResponse
    json_response = JSONResponse(content=result)
    # 复制Cookie设置（如果有）
    for header, value in response.headers.items():
        if header.lower() == "set-cookie":
            json_response.headers.append(header, value)
    return json_response


@app.post("/auth/login")
async def login(payload: Dict[str, str]):
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = generate_jwt(user["id"], user["email"])
    plan = get_plan_for_user(user["id"])
    return {
        "access_token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "plan": plan
        }
    }


@app.get("/auth/me")
async def auth_me(current_user=Depends(get_current_user)):
    plan = get_plan_for_user(current_user["id"])
    return {"id": current_user["id"], "email": current_user["email"], "plan": plan}


@app.post("/auth/logout")
async def logout(current_user=Depends(get_current_user)):
    """登出：将token加入数据库黑名单（商用级改进）"""
    jti = current_user.get("token_jti")
    if jti:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO revoked_tokens (jti, user_id, revoked_at) VALUES (?, ?, ?)",
                (jti, current_user["id"], datetime.utcnow().isoformat()),
            )
    logger.info(f"User logged out: {current_user['email']}")
    return {"success": True}


# -------------------- 支付与订阅 API --------------------
def get_price_id(plan: str) -> str:
    pid = PRICE_IDS.get(plan)
    if not pid:
        raise HTTPException(status_code=400, detail="Price ID not configured for this plan")
    return pid


@app.post("/payments/create-checkout")
async def create_checkout_session(payload: Dict[str, str], current_user=Depends(get_current_user)):
    if not STRIPE_API_KEY:
        raise HTTPException(status_code=500, detail="Stripe API key not configured")

    plan = payload.get("plan", "").lower()
    if plan not in ("starter", "pro"):
        raise HTTPException(status_code=400, detail="Unsupported plan")

    price_id = get_price_id(plan)
    success_url = os.environ.get("STRIPE_SUCCESS_URL", "https://examfrompdf.com/success")
    cancel_url = os.environ.get("STRIPE_CANCEL_URL", "https://examfrompdf.com/cancel")

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=cancel_url,
        customer_email=current_user["email"],
        metadata={"user_id": current_user["id"], "plan": plan},
    )
    return {"checkout_url": session.url}


def upsert_subscription(
    user_id: int,
    plan: str,
    customer_id: str,
    subscription_id: str,
    status: str,
    current_period_end: Optional[int],
):
    """更新或插入订阅信息（使用INSERT OR REPLACE确保数据一致性）"""
    with get_db() as conn:
        cur = conn.cursor()
        # 如果已存在相同subscription_id，则更新；否则插入
        cur.execute(
            """
            INSERT INTO subscriptions (user_id, plan, stripe_customer_id, stripe_subscription_id, status, current_period_end)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(stripe_subscription_id) DO UPDATE SET
                plan = excluded.plan,
                status = excluded.status,
                current_period_end = excluded.current_period_end
            """,
            (user_id, plan, customer_id, subscription_id, status, current_period_end),
        )
    logger.info(f"Subscription updated: user_id={user_id}, plan={plan}, status={status}")


@app.post("/payments/webhook")
async def stripe_webhook(request: Request):
    """
    处理 Stripe Webhook 事件
    - checkout.session.completed: 解锁下载权限（订阅或一次性支付）
    - customer.subscription.updated/deleted: 更新订阅状态
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid Stripe webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    logger.info(f"Received Stripe webhook event: {event_type}")

    # 处理 checkout.session.completed 事件
    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        session_id = session.get("id")
        metadata = session.get("metadata", {})
        payment_type = metadata.get("type")
        
        # 处理订阅支付
        if payment_type != "download_purchase":
            user_id = metadata.get("user_id")
            plan = metadata.get("plan", "starter")
            sub_id = session.get("subscription")
            customer_id = session.get("customer")
            status = "active"
            current_period_end = None
            if sub_id:
                sub_obj = stripe.Subscription.retrieve(sub_id)
                current_period_end = sub_obj.get("current_period_end")
                status = sub_obj.get("status", status)
            if user_id:
                upsert_subscription(int(user_id), plan, customer_id, sub_id, status, current_period_end)
        
        # 处理下载解锁支付
        elif payment_type == "download_purchase":
            job_id = metadata.get("job_id")
            user_id = metadata.get("user_id")
            device_fingerprint = metadata.get("device_fingerprint")
            
            if not job_id:
                logger.error(f"Missing job_id in webhook metadata: {metadata}")
                return {"received": True}
            
            logger.info(f"Processing download purchase webhook: job_id={job_id}, user_id={user_id}, session_id={session_id}")
            
            try:
                with get_db() as conn:
                    cur = conn.cursor()
                    
                    # 验证 job 是否存在
                    cur.execute("SELECT id, user_id, device_fingerprint FROM jobs WHERE id = ?", (job_id,))
                    job_row = cur.fetchone()
                    
                    if not job_row:
                        logger.error(f"Job {job_id} not found in database")
                        return {"received": True}
                    
                    # 验证用户匹配（如果提供了 user_id 或 device_fingerprint）
                    if user_id and job_row["user_id"] is not None:
                        if int(job_row["user_id"]) != int(user_id):
                            logger.warning(f"User ID mismatch: webhook user_id={user_id}, job user_id={job_row['user_id']}")
                    
                    if device_fingerprint and job_row["device_fingerprint"]:
                        if job_row["device_fingerprint"] != device_fingerprint:
                            logger.warning(f"Device fingerprint mismatch for job {job_id}")
                    
                    # 解锁 job
                    unlocked_at = datetime.utcnow().isoformat()
                    cur.execute(
                        "UPDATE jobs SET is_unlocked = 1, unlocked_at = ? WHERE id = ?",
                        (unlocked_at, job_id)
                    )
                    
                    # 更新交易记录状态
                    amount = session.get("amount_total", 99)  # 默认 $0.99
                    currency = session.get("currency", "usd")
                    completed_at = datetime.utcnow().isoformat()
                    cur.execute(
                        """
                        UPDATE transactions 
                        SET status = 'completed', completed_at = ?, amount = ?, currency = ?
                        WHERE stripe_session_id = ?
                        """,
                        (completed_at, amount, currency, session_id)
                    )
                    
                    logger.info(f"Successfully unlocked job {job_id} via webhook")
                    
            except Exception as e:
                logger.error(f"Error processing download purchase webhook: {e}", exc_info=True)
                # 不抛出异常，避免 Stripe 重试

    # 处理订阅更新/删除事件
    elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        sub = event["data"]["object"]
        user_id = sub.get("metadata", {}).get("user_id")
        plan = sub.get("metadata", {}).get("plan", "starter")
        if user_id:
            upsert_subscription(
                int(user_id),
                plan,
                sub.get("customer"),
                sub.get("id"),
                sub.get("status", "active"),
                sub.get("current_period_end"),
            )

    return {"received": True}


@app.get("/payments/unlock-status/{job_id}")
async def get_unlock_status(
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    request: Request = None
):
    """
    检查某个 job 是否已解锁下载
    
    返回：
    {
        "unlocked": true/false,
        "unlocked_at": "2024-01-01T00:00:00" (如果已解锁)
    }
    """
    try:
        # 验证 job 是否存在且属于当前用户/设备
        with get_db() as conn:
            cur = conn.cursor()
            if current_user:
                # 认证用户：验证 job 属于该用户
                cur.execute(
                    "SELECT id, user_id, is_unlocked, unlocked_at FROM jobs WHERE id = ?",
                    (job_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Job not found")
                # 如果 job 有 user_id，必须匹配
                if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
            else:
                # 匿名用户：验证 job 属于该设备
                device_fingerprint = get_device_fingerprint(request)
                cur.execute(
                    "SELECT id, user_id, device_fingerprint, is_unlocked, unlocked_at FROM jobs WHERE id = ?",
                    (job_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Job not found")
                # 如果 job 有 user_id，匿名用户不能访问
                if row["user_id"] is not None:
                    raise HTTPException(status_code=403, detail="Access denied: this job belongs to a registered user. Please log in.")
                # 验证设备指纹
                if row["device_fingerprint"] != device_fingerprint:
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another device")
            
            is_unlocked = bool(row["is_unlocked"]) if row["is_unlocked"] else False
            unlocked_at = row["unlocked_at"] if row["unlocked_at"] else None
            
            return {
                "unlocked": is_unlocked,
                "unlocked_at": unlocked_at
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_unlock_status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/payments/subscription")
async def get_subscription(current_user=Depends(get_current_user)):
    sub = get_active_subscription(current_user["id"])
    if not sub:
        return {"plan": "free", "status": "inactive"}
    return {
        "plan": sub["plan"],
        "status": sub["status"],
        "current_period_end": sub["current_period_end"],
    }


@app.post("/payments/purchase-download")
async def purchase_download(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """
    购买下载权限的支付端点（支持匿名用户和认证用户）
    固定价格 $0.99
    
    请求体：
    - job_id: 要购买的job ID（必需）
    
    返回：
    - Mock模式: { "success": true, "unlocked": true }
    - 正常模式: { "checkout_url": "..." }
    """
    try:
        job_id = payload.get("job_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")
        
        user_info = f"user_id={current_user['id']}" if current_user else "anonymous"
        logger.info(f"[PURCHASE-DOWNLOAD] Request from {user_info}, job_id={job_id}")
        
        # 验证 job 是否存在且属于当前用户/设备
        with get_db() as conn:
            cur = conn.cursor()
            if current_user:
                # 认证用户：验证 job 属于该用户
                cur.execute(
                    "SELECT id, user_id, device_fingerprint, status, is_unlocked FROM jobs WHERE id = ?",
                    (job_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Job not found")
                # 如果 job 有 user_id，必须匹配
                if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
            else:
                # 匿名用户：验证 job 属于该设备
                device_fingerprint = get_device_fingerprint(request)
                cur.execute(
                    "SELECT id, user_id, device_fingerprint, status, is_unlocked FROM jobs WHERE id = ?",
                    (job_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Job not found")
                # 如果 job 有 user_id，匿名用户不能访问
                if row["user_id"] is not None:
                    raise HTTPException(status_code=403, detail="Access denied: this job belongs to a registered user. Please log in.")
                # 验证设备指纹
                if row["device_fingerprint"] != device_fingerprint:
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another device")
            
            # 检查 job 是否已完成
            if row["status"] != "done":
                raise HTTPException(status_code=400, detail="Job is not completed yet")
            
            # 检查是否已经解锁
            if row["is_unlocked"]:
                logger.info(f"Job {job_id} is already unlocked for {user_info}")
                return {"success": True, "unlocked": True, "message": "Job is already unlocked"}
        
        # Mock 模式：直接返回成功并解锁
        if ENABLE_MOCK_PAYMENT:
            logger.info(f"Mock payment enabled: granting download access for {user_info}, job_id={job_id}")
            # 在 Mock 模式下直接解锁
            with get_db() as conn:
                cur = conn.cursor()
                unlocked_at = datetime.utcnow().isoformat()
                cur.execute(
                    "UPDATE jobs SET is_unlocked = 1, unlocked_at = ? WHERE id = ?",
                    (unlocked_at, job_id)
                )
            return {"success": True, "unlocked": True}
        
        # 正常模式：创建 Stripe Checkout Session
        if not STRIPE_SECRET_KEY:
            # 如果没有配置 Stripe 且不在生产环境，自动启用 Mock 模式
            if not IS_PRODUCTION:
                logger.warning("Stripe API key not configured, but not in production. Using mock mode as fallback.")
                logger.info(f"Mock payment (fallback): unlocking job {job_id} for {user_info}")
                # Mock 模式下直接解锁
                with get_db() as conn:
                    cur = conn.cursor()
                    unlocked_at = datetime.utcnow().isoformat()
                    cur.execute(
                        "UPDATE jobs SET is_unlocked = 1, unlocked_at = ? WHERE id = ?",
                        (unlocked_at, job_id)
                    )
                return {"success": True, "unlocked": True}
            else:
                logger.error("Stripe API key not configured in production environment")
                raise HTTPException(status_code=500, detail="Stripe API key not configured")
        
        # 固定价格 $0.99 (99 美分)
        amount = 99
        
        # 构建成功和取消URL
        base_url = os.environ.get("API_BASE_URL", "https://examfrompdf.com").rstrip('/')
        success_url = f"{base_url}/?payment=success&job_id={job_id}"
        cancel_url = f"{base_url}/?payment=cancelled"
        
        logger.info(f"Creating Stripe checkout session: amount=${amount/100:.2f}, job_id={job_id}, {user_info}")
        
        # 创建一次性支付的 Checkout Session
        try:
            # 准备 metadata
            metadata = {
                "type": "download_purchase",
                "job_id": job_id,
            }
            
            # 如果是认证用户，添加 user_id；如果是匿名用户，添加 device_fingerprint
            if current_user:
                metadata["user_id"] = str(current_user["id"])
                customer_email = current_user["email"]
            else:
                device_fingerprint = get_device_fingerprint(request)
                metadata["device_fingerprint"] = device_fingerprint
                customer_email = None  # 匿名用户可能没有邮箱
            
            # 构建 line_items：优先使用 STRIPE_PRICE_ID，否则使用 price_data
            # 检查 Price ID 是否有效（不是占位符）
            if STRIPE_PRICE_ID and STRIPE_PRICE_ID != "price_xxxxx" and STRIPE_PRICE_ID.startswith("price_"):
                line_items = [{
                    "price": STRIPE_PRICE_ID,
                    "quantity": 1,
                }]
                logger.info(f"Using Stripe Price ID: {STRIPE_PRICE_ID}")
            else:
                # 使用 price_data 动态创建价格
                line_items = [{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": "Exam Download Access",
                            "description": f"Download access for exam {job_id}",
                        },
                        "unit_amount": amount,  # 金额（美分）
                    },
                    "quantity": 1,
                }]
                if STRIPE_PRICE_ID:
                    logger.warning(f"Invalid or placeholder Price ID '{STRIPE_PRICE_ID}' detected, using price_data instead")
                else:
                    logger.info("No STRIPE_PRICE_ID configured, using price_data")
            
            session_params = {
                "mode": "payment",  # 一次性支付，不是订阅
                "line_items": line_items,
                "success_url": success_url,
                "cancel_url": cancel_url,
                "metadata": metadata,
            }
            
            # 只有认证用户才设置 customer_email
            if customer_email:
                session_params["customer_email"] = customer_email
            
            session = stripe.checkout.Session.create(**session_params)
            
            # 保存交易记录（状态为 pending）
            with get_db() as conn:
                cur = conn.cursor()
                created_at = datetime.utcnow().isoformat()
                user_id_val = current_user["id"] if current_user else None
                cur.execute(
                    """
                    INSERT INTO transactions (job_id, user_id, stripe_session_id, amount, currency, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, user_id_val, session.id, amount, "usd", "pending", created_at)
                )
            
            logger.info(f"Created checkout session {session.id} for {user_info}, amount: ${amount/100:.2f}")
            return {"checkout_url": session.url}
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating checkout session: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Payment processing error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in purchase_download: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _generate_preview_image(job_id: str, pdf_path: Path, job_dir: Path):
    """
    生成PDF第一页的预览图（带水印）
    
    参数:
    - job_id: 任务ID
    - pdf_path: PDF文件路径
    - job_dir: job目录路径
    
    生成的文件:
    - job_dir/build/preview.png (800x1100px, A4比例)
    """
    if not PDF_PREVIEW_AVAILABLE:
        logger.warning("PDF preview libraries not available, skipping preview generation")
        return
    
    try:
        # 打开PDF文件
        pdf_document = fitz.open(pdf_path)
        
        if len(pdf_document) == 0:
            logger.warning(f"PDF {pdf_path} has no pages, cannot generate preview")
            pdf_document.close()
            return
        
        # 获取第一页
        first_page = pdf_document[0]
        
        # 计算缩放比例以得到800x1100的尺寸（A4比例）
        # A4尺寸: 210mm x 297mm，比例约为 1:1.414
        # 目标尺寸: 800x1100 (比例约为 1:1.375，接近A4)
        target_width = 800
        target_height = 1100
        
        # 获取原始页面尺寸
        page_rect = first_page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # 计算缩放比例（保持宽高比，以宽度为准）
        zoom_x = target_width / page_width
        zoom_y = target_height / page_height
        zoom = min(zoom_x, zoom_y)  # 取较小的值，确保不超出目标尺寸
        
        # 设置矩阵（缩放）
        mat = fitz.Matrix(zoom, zoom)
        
        # 渲染页面为图片（DPI会影响质量，这里使用缩放矩阵）
        pix = first_page.get_pixmap(matrix=mat, alpha=False)
        
        # 转换为PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # 如果图片尺寸不是目标尺寸，进行裁剪或填充
        if img.size != (target_width, target_height):
            # 创建目标尺寸的白色背景
            preview_img = Image.new("RGB", (target_width, target_height), "white")
            # 计算居中位置
            x_offset = (target_width - img.width) // 2
            y_offset = (target_height - img.height) // 2
            # 粘贴图片到中心
            preview_img.paste(img, (x_offset, y_offset))
            img = preview_img
        
        # 添加水印
        draw = ImageDraw.Draw(img)
        
        # 尝试使用系统字体，如果失败则使用默认字体
        try:
            # 尝试使用较大的字体
            font_size = 32
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            except:
                # 使用默认字体
                font = ImageFont.load_default()
        
        # 水印文字
        watermark_text = "PREVIEW - Pay to Download Full Exam"
        
        # 计算文字位置（居中，稍微偏下）
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (target_width - text_width) // 2
        y = target_height - text_height - 50  # 距离底部50像素
        
        # 添加半透明背景框（让文字更清晰）
        padding = 10
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 180)  # 半透明黑色背景
        )
        img = Image.alpha_composite(img.convert("RGBA"), overlay)
        
        # 重新创建draw对象（因为img现在是RGBA模式）
        draw = ImageDraw.Draw(img)
        
        # 绘制文字阴影（深灰色，稍微偏移）
        draw.text((x + 2, y + 2), watermark_text, fill=(50, 50, 50, 255), font=font)
        # 绘制文字（白色）
        draw.text((x, y), watermark_text, fill="white", font=font)
        
        # 转换回RGB模式
        img = img.convert("RGB")
        
        # 保存预览图
        preview_path = job_dir / "build" / "preview.png"
        img.save(preview_path, "PNG", optimize=True)
        
        logger.info(f"Preview image generated successfully for job {job_id} at {preview_path}")
        
        # 关闭PDF文档
        pdf_document.close()
        
    except Exception as e:
        logger.error(f"Failed to generate preview image for job {job_id}: {e}", exc_info=True)
        raise


def _generate_answer_for_job_if_missing(job_id: str, user_id: int) -> bool:
    """
    为指定的job生成答案（如果缺失）
    
    返回：
    - True: 成功生成答案或答案已存在
    - False: 生成失败（但不抛出异常）
    """
    try:
        job_dir = BUILD_ROOT / job_id
        exam_data_path = job_dir / "exam_data.json"
        
        if not exam_data_path.exists():
            logger.warning(f"exam_data.json not found for job {job_id}, cannot generate answer")
            return False
        
        # 读取exam_data
        import json
        with open(exam_data_path, "r", encoding="utf-8") as f:
            exam_data = json.load(f)
        
        # 如果已经有答案，检查答案PDF是否存在
        if "answers" in exam_data:
            answer_pdf_path = job_dir / "build" / "answer_filled.pdf"
            if answer_pdf_path.exists():
                logger.info(f"Answer already exists for job {job_id}")
                return True
            else:
                # 有答案数据但没有PDF，重新生成PDF
                logger.info(f"Answer data exists but PDF missing for job {job_id}, regenerating PDF")
                _render_and_compile_answer_pdf(job_id, job_dir, exam_data_path)
                return True
        
        # 没有答案数据，生成答案
        logger.info(f"Generating answer for old job {job_id} (user {user_id})")
        
        # 导入generate_answer_key函数
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from generate_exam_data import generate_answer_key
        
        # 生成答案
        answer_key = generate_answer_key(exam_data)
        exam_data["answers"] = answer_key
        
        # 保存更新后的exam_data.json
        with open(exam_data_path, "w", encoding="utf-8") as f:
            json.dump(exam_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Answer key generated and saved for job {job_id}")
        
        # 生成答案PDF
        _render_and_compile_answer_pdf(job_id, job_dir, exam_data_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate answer for job {job_id}: {e}", exc_info=True)
        return False


def _render_and_compile_answer_pdf(job_id: str, job_dir: Path, exam_data_path: Path):
    """渲染并编译答案PDF"""
    try:
        # 渲染答案LaTeX
        env = os.environ.copy()
        env["EXAMGEN_OUTPUT_DIR"] = str(job_dir / "build")
        env["EXAMGEN_EXAM_DATA"] = str(exam_data_path)
        subprocess.run(
            [sys.executable, str(BASE_DIR / "render_answer.py")],
            cwd=str(BASE_DIR),
            check=True,
            env=env,
        )
        
        # 编译答案PDF
        answer_tex_path = job_dir / "build" / "answer_filled.tex"
        if answer_tex_path.exists():
            # 检测是否需要中文支持
            with open(answer_tex_path, "r", encoding="utf-8") as f:
                answer_tex_content = f.read()
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in answer_tex_content)
            compiler = "xelatex" if has_chinese else "pdflatex"
            
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode", "-output-directory", str(job_dir / "build"), str(answer_tex_path)],
                capture_output=True,
                text=True,
            )
            
            answer_pdf_path = job_dir / "build" / "answer_filled.pdf"
            if answer_pdf_path.exists():
                logger.info(f"Answer PDF generated successfully for job {job_id}")
            else:
                logger.warning(f"Answer PDF was not generated for job {job_id} (exit code: {result.returncode})")
    except Exception as e:
        logger.error(f"Failed to render/compile answer PDF for job {job_id}: {e}", exc_info=True)
        raise


# -------------------- 用量 API --------------------
@app.get("/usage/status")
async def usage_status(current_user=Depends(get_current_user)):
    plan = get_plan_for_user(current_user["id"])
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    daily_count, monthly_count = get_usage_counts(current_user["id"])
    
    daily_limit = limits.get("daily", 1)
    monthly_limit = limits.get("monthly", 30)
    
    # 计算是否可以生成（已移除注册奖励逻辑）
    if plan == "free":
        can_generate = daily_count < daily_limit
    else:
        can_generate = monthly_count < monthly_limit
    
    return {
        "plan": plan,
        "daily_used": daily_count,
        "daily_limit": daily_limit,
        "monthly_used": monthly_count,
        "monthly_limit": monthly_limit,
        "can_generate": can_generate,
    }


# -------------------- 监控统计 API --------------------
@app.get("/monitoring/stats")
async def get_monitoring_stats(current_user=Depends(get_current_user)):
    """
    获取系统监控统计信息，包括用户等待时长统计
    返回格式：
    {
        "user_stats": {
            "total_jobs": 10,
            "completed_jobs": 8,
            "failed_jobs": 1,
            "queued_jobs": 1,
            "avg_wait_time_seconds": 5.2,  # 平均队列等待时间（从创建到开始处理）
            "avg_processing_time_seconds": 45.3,  # 平均处理时间（从开始到完成）
            "avg_total_time_seconds": 50.5,  # 平均总时间（从创建到完成）
            "max_wait_time_seconds": 15.8,
            "max_processing_time_seconds": 120.5,
            "max_total_time_seconds": 135.2
        },
        "recent_jobs": [
            {
                "job_id": "...",
                "wait_time_seconds": 5.2,
                "processing_time_seconds": 45.3,
                "total_time_seconds": 50.5,
                "status": "done"
            }
        ]
    }
    """
    with get_db() as conn:
        cur = conn.cursor()
        
        # 获取用户的所有job
        cur.execute(
            """
            SELECT id, status, created_at, started_at, completed_at
            FROM jobs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (current_user["id"],)
        )
        rows = cur.fetchall()
        
        # 统计信息
        total_jobs = len(rows)
        completed_jobs = [r for r in rows if r["status"] == "done"]
        failed_jobs = [r for r in rows if r["status"] == "failed"]
        queued_jobs = [r for r in rows if r["status"] in ["queued", "running"]]
        
        # 计算等待时间、处理时间和总时间
        wait_times = []
        processing_times = []
        total_times = []
        recent_jobs = []
        
        for row in rows[:20]:  # 只统计最近20个job
            job_id = row["id"]
            status = row["status"]
            created_at_str = row["created_at"]
            started_at_str = row["started_at"]
            completed_at_str = row["completed_at"]
            
            if not created_at_str:
                continue
            
            # 解析时间字符串（兼容有无时区的情况）
            def parse_datetime(dt_str):
                if not dt_str:
                    return None
                # 移除Z并尝试解析
                dt_str = dt_str.replace('Z', '')
                try:
                    # 尝试解析带时区的格式
                    return datetime.fromisoformat(dt_str)
                except ValueError:
                    # 如果没有时区，假设是UTC
                    try:
                        return datetime.fromisoformat(dt_str + '+00:00')
                    except ValueError:
                        # 最后尝试基本格式
                        return datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f')
            
            created_at = parse_datetime(created_at_str)
            if not created_at:
                continue
            
            job_stats = {
                "job_id": job_id,
                "status": status,
                "wait_time_seconds": None,
                "processing_time_seconds": None,
                "total_time_seconds": None
            }
            
            # 计算队列等待时间（从创建到开始处理）
            if started_at_str:
                started_at = parse_datetime(started_at_str)
                if started_at:
                    wait_time = (started_at - created_at).total_seconds()
                    if wait_time >= 0:  # 确保时间差是合理的
                        wait_times.append(wait_time)
                        job_stats["wait_time_seconds"] = round(wait_time, 2)
            
            # 计算处理时间（从开始到完成）
            if started_at_str and completed_at_str:
                started_at = parse_datetime(started_at_str)
                completed_at = parse_datetime(completed_at_str)
                if started_at and completed_at:
                    processing_time = (completed_at - started_at).total_seconds()
                    if processing_time >= 0:
                        processing_times.append(processing_time)
                        job_stats["processing_time_seconds"] = round(processing_time, 2)
            
            # 计算总时间（从创建到完成）
            if completed_at_str:
                completed_at = parse_datetime(completed_at_str)
                if completed_at:
                    total_time = (completed_at - created_at).total_seconds()
                    if total_time >= 0:
                        total_times.append(total_time)
                        job_stats["total_time_seconds"] = round(total_time, 2)
            
            recent_jobs.append(job_stats)
        
        # 计算平均值和最大值
        avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_total_time = sum(total_times) / len(total_times) if total_times else 0
        
        max_wait_time = max(wait_times) if wait_times else 0
        max_processing_time = max(processing_times) if processing_times else 0
        max_total_time = max(total_times) if total_times else 0
        
        return {
            "user_stats": {
                "total_jobs": total_jobs,
                "completed_jobs": len(completed_jobs),
                "failed_jobs": len(failed_jobs),
                "queued_jobs": len(queued_jobs),
                "avg_wait_time_seconds": round(avg_wait_time, 2),
                "avg_processing_time_seconds": round(avg_processing_time, 2),
                "avg_total_time_seconds": round(avg_total_time, 2),
                "max_wait_time_seconds": round(max_wait_time, 2),
                "max_processing_time_seconds": round(max_processing_time, 2),
                "max_total_time_seconds": round(max_total_time, 2)
            },
            "recent_jobs": recent_jobs
        }


# -------------------- 任务历史 API --------------------
@app.get("/jobs")
async def get_jobs(current_user=Depends(get_current_user), limit: int = 50, offset: int = 0):
    """
    获取用户的历史任务列表（商用级：完全从数据库读取）
    返回格式：
    {
        "jobs": [
            {
                "id": "job_id",
                "jobId": "job_id",
                "fileName": "lecture.pdf",
                "status": "done",
                "createdAt": "2026-01-21T10:00:00",
                "downloadUrl": "/download/job_id",
                "error": null
            }
        ],
        "total": 10
    }
    """
    logger.info(f"[GET JOBS] Request from user_id={current_user['id']}, limit={limit}, offset={offset}")
    with get_db() as conn:
        cur = conn.cursor()
        # 先获取总数（包括user_id匹配的，以及user_id为NULL但device_fingerprint匹配的）
        # 注意：这里只统计user_id匹配的，因为匿名用户的job不应该出现在认证用户的历史中
        cur.execute("SELECT COUNT(*) as total FROM jobs WHERE user_id = ?", (current_user["id"],))
        total = cur.fetchone()["total"]
        logger.info(f"[GET JOBS] Found {total} total jobs for user_id={current_user['id']}")
        
        # 获取任务列表（只返回user_id匹配的job）
        # 如果job的user_id为NULL，说明是匿名用户创建的，不应该出现在认证用户的历史中
        cur.execute(
            """
            SELECT id, file_name, status, created_at, download_url, error
            FROM jobs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (current_user["id"], limit, offset),
        )
        rows = cur.fetchall()
        logger.info(f"[GET JOBS] Returning {len(rows)} jobs for user_id={current_user['id']}")
        
        # 记录返回的job IDs用于调试
        if rows:
            job_ids = [row["id"] for row in rows]
            logger.info(f"[GET JOBS] Job IDs being returned: {job_ids[:5]}{'...' if len(job_ids) > 5 else ''}")
    
    jobs = []
    for row in rows:
        job = {
            "id": row["id"],
            "jobId": row["id"],  # 兼容前端字段名
            "fileName": row["file_name"],
            "status": row["status"],
            "createdAt": row["created_at"],
        }
        if row["download_url"]:
            job["downloadUrl"] = row["download_url"]
        if row["error"]:
            job["error"] = row["error"]
        
        jobs.append(job)
        logger.info(f"[GET JOBS] Added job {row['id'][:8]}... with file_name='{row['file_name']}' to response")
    
    logger.info(f"[GET JOBS] Final response: {len(jobs)} jobs, total={total}")
    # 记录所有返回的 file_name 用于调试
    file_names = [job['fileName'] for job in jobs]
    logger.info(f"[GET JOBS] Returning fileNames: {file_names}")
    return {"jobs": jobs, "total": total}


@app.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    删除指定的 job（支持匿名用户和认证用户）
    - 验证 job 属于当前用户/设备
    - 删除数据库记录
    - 删除对应的文件目录
    """
    try:
        logger.info(f"[DELETE JOB] Request to delete job {job_id}")
        
        # 验证 job 是否存在且属于当前用户/设备
        job_exists = False
        user_info = ""
        
        with get_db() as conn:
            cur = conn.cursor()
            if current_user:
                # 认证用户：验证 job 属于该用户
                cur.execute(
                    "SELECT id, user_id, device_fingerprint, file_name FROM jobs WHERE id = ?",
                    (job_id,)
                )
                row = cur.fetchone()
                if not row:
                    # Job 不存在（可能已经被删除）- 幂等性：返回成功
                    logger.info(f"[DELETE JOB] Job {job_id} not found, already deleted (user_id={current_user['id']})")
                    return {"message": "Job already deleted or not found", "job_id": job_id}
                # 如果 job 有 user_id，必须匹配
                if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
                job_exists = True
                user_info = f"user_id={current_user['id']}"
            else:
                # 匿名用户：验证 job 属于该设备
                device_fingerprint = get_device_fingerprint(request)
                cur.execute(
                    "SELECT id, user_id, device_fingerprint, file_name FROM jobs WHERE id = ?",
                    (job_id,)
                )
                row = cur.fetchone()
                if not row:
                    # Job 不存在（可能已经被删除）- 幂等性：返回成功
                    logger.info(f"[DELETE JOB] Job {job_id} not found, already deleted (anonymous)")
                    return {"message": "Job already deleted or not found", "job_id": job_id}
                # 如果 job 有 user_id，匿名用户不能删除
                if row["user_id"] is not None:
                    raise HTTPException(status_code=403, detail="Access denied: this job belongs to a registered user. Please log in.")
                # 验证设备指纹
                if row["device_fingerprint"] != device_fingerprint:
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another device")
                job_exists = True
                user_info = "anonymous"
            
            # 删除数据库记录
            cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            deleted_count = cur.rowcount
            logger.info(f"[DELETE JOB] DELETE query executed, rowcount={deleted_count} for job {job_id}")
            
            if deleted_count == 0:
                # 这种情况理论上不应该发生（因为上面已经检查了 row），但为了安全起见
                logger.warning(f"[DELETE JOB] Job {job_id} was found but deletion returned 0 rows")
                # 幂等性：返回成功
                return {"message": "Job already deleted or not found", "job_id": job_id}
            
            # 验证删除是否成功（在事务提交前）
            cur.execute("SELECT COUNT(*) as count FROM jobs WHERE id = ?", (job_id,))
            verify_row = cur.fetchone()
            if verify_row["count"] > 0:
                logger.error(f"[DELETE JOB] CRITICAL: Job {job_id} still exists after DELETE query!")
            else:
                logger.info(f"[DELETE JOB] Verified: Job {job_id} successfully deleted from database (before commit)")
        
        # 事务会在退出 with 块时自动提交（get_db 上下文管理器）
        logger.info(f"[DELETE JOB] Database transaction will be committed for job {job_id}")
        
        # 删除文件目录
        job_dir = BUILD_ROOT / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info(f"[DELETE JOB] Deleted files for job {job_id} ({user_info})")
            except Exception as e:
                logger.warning(f"[DELETE JOB] Failed to delete files for job {job_id}: {e}", exc_info=True)
                # 文件删除失败不影响数据库删除，继续执行
        
        # 删除关联的交易记录（如果有）
        try:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM transactions WHERE job_id = ?", (job_id,))
                logger.info(f"[DELETE JOB] Deleted transaction records for job {job_id}")
        except Exception as e:
            logger.warning(f"[DELETE JOB] Failed to delete transaction records for job {job_id}: {e}", exc_info=True)
            # 交易记录删除失败不影响主流程
        
        # 最终验证：再次查询数据库确认记录已删除（事务已提交后）
        try:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) as count FROM jobs WHERE id = ?", (job_id,))
                final_verify = cur.fetchone()
                if final_verify["count"] > 0:
                    logger.error(f"[DELETE JOB] CRITICAL: Job {job_id} still exists in database after commit!")
                else:
                    logger.info(f"[DELETE JOB] Final verification: Job {job_id} confirmed deleted from database")
        except Exception as e:
            logger.warning(f"[DELETE JOB] Could not verify final deletion: {e}")
        
        logger.info(f"[DELETE JOB] Successfully deleted job {job_id} ({user_info})")
        return {"message": "Job deleted successfully", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DELETE JOB] Unexpected error deleting job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


def run_job(job_id: str, lecture_paths: List[Path], exam_config: Optional[Dict[str, Any]] = None):
    """
    后台执行：PDF(s) -> exam_data.json -> render_exam.py -> pdflatex -> PDF
    结果写入数据库（商用级改进：移除内存状态）
    
    参数：
    - job_id: 任务ID
    - lecture_paths: PDF文件路径列表（支持多个文件）
    - exam_config: 包含 mcq_count, short_answer_count, long_question_count, difficulty, special_requests
    """
    # 设置默认值
    if exam_config is None:
        exam_config = {
            "mcq_count": 10,
            "short_answer_count": 3,
            "long_question_count": 1,
            "difficulty": "medium",
            "special_requests": None
        }
    
    job_dir = BUILD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 处理多个PDF文件：确保所有文件都在job_dir中
    job_lecture_paths = []
    for idx, lecture_path in enumerate(lecture_paths):
        if len(lecture_paths) == 1:
            job_lecture = job_dir / "lecture.pdf"
        else:
            job_lecture = job_dir / f"lecture_{idx}.pdf"
        
        # 如果文件不在job_dir中，复制它
        if lecture_path != job_lecture:
            shutil.copy2(lecture_path, job_lecture)
        
        job_lecture_paths.append(job_lecture)

    # 统一输出在 job_dir/build 下
    (job_dir / "build").mkdir(exist_ok=True)

    try:
        # 记录开始处理时间
        started_at = datetime.utcnow().isoformat()
        
        # 更新数据库状态（商用级：移除内存状态）
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs SET status = ?, updated_at = ?, started_at = ? WHERE id = ?
                """,
                ("running", started_at, started_at, job_id),
            )
        logger.info(f"Job {job_id} started processing at {started_at}")

        # 1) 生成 JSON - 直接输出到 job_dir，避免并发冲突
        job_exam_data_json = job_dir / "exam_data.json"
        progress_file = job_dir / "progress.json"
        env = os.environ.copy()
        # 传递exam配置参数到generate_exam_data.py
        env["EXAMGEN_MCQ_COUNT"] = str(exam_config["mcq_count"])
        env["EXAMGEN_SHORT_ANSWER_COUNT"] = str(exam_config["short_answer_count"])
        env["EXAMGEN_LONG_QUESTION_COUNT"] = str(exam_config["long_question_count"])
        env["EXAMGEN_DIFFICULTY"] = exam_config["difficulty"]
        if exam_config.get("special_requests"):
            env["EXAMGEN_SPECIAL_REQUESTS"] = exam_config["special_requests"]
        logger.info(f"Passing exam config to generate_exam_data.py: MCQ={exam_config['mcq_count']}, SAQ={exam_config['short_answer_count']}, LQ={exam_config['long_question_count']}, Difficulty={exam_config['difficulty']}, SpecialRequests={exam_config.get('special_requests', 'None')[:50] if exam_config.get('special_requests') else 'None'}..., PDF files: {len(job_lecture_paths)}")
        # 构建命令行参数：所有PDF路径 + 输出JSON路径
        cmd = [sys.executable, str(BASE_DIR / "generate_exam_data.py")]
        cmd.extend([str(path) for path in job_lecture_paths])  # 添加所有PDF路径
        cmd.append(str(job_exam_data_json))  # 添加输出JSON路径
        
        # 启动子进程并定期读取进度
        import threading
        progress_updated = threading.Event()
        
        def update_progress_from_file():
            """定期读取进度文件并更新数据库"""
            while not progress_updated.is_set():
                try:
                    if progress_file.exists():
                        with open(progress_file, "r", encoding="utf-8") as f:
                            progress_data = json.load(f)
                        # 更新数据库
                        with get_db() as conn:
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE jobs SET progress_info = ? WHERE id = ?",
                                (json.dumps(progress_data), job_id)
                            )
                except Exception as e:
                    # 进度更新失败不影响主流程
                    pass
                time.sleep(2)  # 每2秒更新一次
        
        # 启动进度更新线程
        progress_thread = threading.Thread(target=update_progress_from_file, daemon=True)
        progress_thread.start()
        
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            check=False,  # 不自动抛出异常，我们自己处理
            env=env,
            capture_output=True,
            text=True,
        )
        
        # 停止进度更新线程
        progress_updated.set()
        
        if result.returncode != 0:
            error_output = result.stderr if result.stderr else result.stdout
            logger.error(f"generate_exam_data.py failed with exit code {result.returncode}")
            logger.error(f"Command: {' '.join(cmd)}")
            if error_output:
                logger.error(f"Error output (last 2000 chars): {error_output[-2000:]}")
            raise RuntimeError(f"generate_exam_data.py failed (exit code {result.returncode}): {error_output[-500:] if error_output else 'No error output'}")

        # 将 templates 拷一份（避免你以后改模板影响旧 job）
        # 可选：如果你不想拷模板，可以不拷，直接用 BASE_DIR/templates
        # 这里不拷，直接复用 BASE_DIR/templates（更轻）

        # 在 job_dir 里运行 render_exam.py：让它输出 job_dir/build/exam_filled.tex
        # 你 render_exam.py 里如果写死 OUTPUT_DIR=BASE_DIR/build，需要改成支持环境变量
        # 为了让你“直接跑通”，我们这里用一个环境变量告诉 render_exam.py 输出到 job_dir/build
        env = os.environ.copy()
        env["EXAMGEN_OUTPUT_DIR"] = str(job_dir / "build")
        env["EXAMGEN_EXAM_DATA"] = str(job_dir / "exam_data.json")
        subprocess.run(
            [sys.executable, str(BASE_DIR / "render_exam.py")],
            cwd=str(BASE_DIR),
            check=True,
            env=env,
        )

        # 3) 编译 PDF - 检测是否需要中文支持
        tex_path = job_dir / "build" / "exam_filled.tex"
        
        # 读取生成的tex文件，检测是否包含中文字符
        with open(tex_path, "r", encoding="utf-8") as f:
            tex_content = f.read()
        
        # 检测是否包含中文字符（CJK字符范围）
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in tex_content)
        
        if has_chinese:
            # 使用xelatex支持中文
            compiler = "xelatex"
        else:
            # 使用pdflatex（更快，兼容性更好）
            compiler = "pdflatex"
        
        # 编译 LaTeX（允许警告，只要PDF生成成功即可）
        result = subprocess.run(
            [compiler, "-interaction=nonstopmode", "-output-directory", str(job_dir / "build"), str(tex_path)],
            capture_output=True,
            text=True,
        )

        pdf_path = job_dir / "build" / "exam_filled.pdf"
        
        # 检查PDF是否成功生成（这是最重要的）
        if not pdf_path.exists():
            # PDF未生成，记录错误输出
            logger.error(f"LaTeX compilation failed. Exit code: {result.returncode}")
            logger.error(f"LaTeX stderr: {result.stderr[-1000:]}")  # 只记录最后1000字符
            raise RuntimeError(f"PDF was not generated. LaTeX exit code: {result.returncode}")
        
        # PDF已生成，即使有警告也视为成功
        if result.returncode != 0:
            # 有警告但PDF生成了，记录警告但不失败
            logger.warning(f"LaTeX compilation completed with warnings (exit code: {result.returncode}), but PDF was generated successfully")
            # 可选：记录警告信息（但不要太多）
            if result.stderr:
                # 只记录关键警告，避免日志过多
                warnings = [line for line in result.stderr.split('\n') if 'Warning' in line or 'Error' in line]
                if warnings:
                    logger.warning(f"LaTeX warnings: {warnings[:5]}")  # 只记录前5个警告

        # 4) 生成预览图（第一页，带水印）
        try:
            _generate_preview_image(job_id, pdf_path, job_dir)
        except Exception as e:
            logger.warning(f"Failed to generate preview image for job {job_id}: {e}", exc_info=True)
            # 预览图生成失败不影响主流程，继续执行

        # 5) 不再自动生成答案PDF，等用户付款后再生成（节省token）
        logger.info(f"Skipping answer generation for job {job_id} (will be generated after payment)")

        # 记录完成时间
        completed_at = datetime.utcnow().isoformat()
        
        # 更新数据库（商用级：移除内存状态）
        download_url = f"/download/{job_id}"
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs SET status = ?, download_url = ?, updated_at = ?, completed_at = ? WHERE id = ?
                """,
                ("done", download_url, completed_at, completed_at, job_id),
            )
        logger.info(f"Job {job_id} completed successfully at {completed_at}")

    except Exception as e:
        error_msg = str(e)
        completed_at = datetime.utcnow().isoformat()
        # 更新数据库（商用级：移除内存状态）
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs SET status = ?, error = ?, updated_at = ?, completed_at = ? WHERE id = ?
                """,
                ("failed", error_msg, completed_at, completed_at, job_id),
            )
        logger.error(f"Job {job_id} failed at {completed_at}: {error_msg}", exc_info=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
      <head><title>ExamFromPDF</title></head>
      <body>
        <h1>ExamFromPDF</h1>
        <form action="/generate" method="post" enctype="multipart/form-data">
          <input type="file" name="lecture_pdf" accept="application/pdf" required />
          <button type="submit">Generate</button>
        </form>
      </body>
    </html>
    """

@app.get("/health")
async def health():
    """健康检查端点（公开，无需认证）"""
    return {"status": "ok"}


@app.get("/queue/status/public")
async def get_queue_status_public():
    """
    获取任务队列状态（公开端点，用于监控，无需认证）
    注意：这是监控端点，不包含敏感信息
    """
    with queue_stats_lock:
        stats = queue_stats.copy()
    
    # 计算预估等待时间（假设每个任务平均2分钟）
    avg_task_time = 120  # 秒
    estimated_wait = stats["current_queue_size"] * avg_task_time / MAX_CONCURRENT_JOBS
    
    return {
        "queue_size": stats["current_queue_size"],
        "max_queue_size": stats["max_queue_size"],
        "processing": stats["current_processing"],
        "max_processing": stats["max_processing"],
        "max_concurrent": MAX_CONCURRENT_JOBS,
        "total_processed": stats["total_processed"],
        "total_failed": stats["total_failed"],
        "estimated_wait_time": int(estimated_wait),
    }


# -------------------- 队列监控 API --------------------
@app.get("/queue/status")
async def get_queue_status(current_user=Depends(get_current_user)):
    """
    获取任务队列状态（阶段1改进：监控功能）
    返回：
    {
        "queue_size": 5,           # 当前排队数
        "max_queue_size": 20,       # 历史最高排队数
        "processing": 3,            # 当前正在处理数
        "max_processing": 5,        # 历史最高同时处理数
        "max_concurrent": 5,        # 最大并发数
        "total_processed": 100,     # 总处理数
        "total_failed": 2,          # 总失败数
        "estimated_wait_time": 120  # 预估等待时间（秒）
    }
    """
    with queue_stats_lock:
        stats = queue_stats.copy()
    
    # 计算预估等待时间（假设每个任务平均2分钟）
    avg_task_time = 120  # 秒
    estimated_wait = stats["current_queue_size"] * avg_task_time / MAX_CONCURRENT_JOBS
    
    return {
        "queue_size": stats["current_queue_size"],
        "max_queue_size": stats["max_queue_size"],
        "processing": stats["current_processing"],
        "max_processing": stats["max_processing"],
        "max_concurrent": MAX_CONCURRENT_JOBS,
        "total_processed": stats["total_processed"],
        "total_failed": stats["total_failed"],
        "estimated_wait_time": int(estimated_wait),
    }


# 文件大小限制：50MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Pydantic 模型
class GenerateExamRequest(BaseModel):
    session_id: str
    file_name: Optional[str] = None  # 可选：前端传递的文件名（用于显示）
    mcq_count: int = 10
    short_answer_count: int = 3
    long_question_count: int = 1
    difficulty: str = "medium"
    special_requests: Optional[str] = None

# -------------------- 文件上传 API --------------------
@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    上传单个PDF文件到指定session
    返回：{ "success": true, "file_name": "...", "session_id": "...", "file_count": N }
    
    请求格式：multipart/form-data
    - file: PDF文件
    - session_id: UUID格式的session ID
    """
    logger.info(f"[UPLOAD] Received upload request: filename={file.filename}, content_type={file.content_type}, session_id={session_id}")
    
    # 验证文件类型
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # 验证session_id格式（UUID格式）
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    # 创建session目录
    session_dir = UPLOAD_SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成唯一文件名（避免重名覆盖）
    file_index = len(list(session_dir.glob("*.pdf")))
    save_path = session_dir / f"file_{file_index:03d}.pdf"
    
    # 保存文件（带超时和大小检查）
    file_size = 0
    start_time = time.time()
    timeout = 60  # 1分钟超时
    
    try:
        with save_path.open("wb") as f:
            chunk_size = 8192  # 8KB chunks
            while True:
                # 检查超时
                if time.time() - start_time > timeout:
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=408, detail="File upload timeout (60s)")
                
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                
                # 检查文件大小
                if file_size > MAX_FILE_SIZE:
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.0f}MB"
                    )
                
                f.write(chunk)
        
        # 统计session中的文件数量
        file_count = len(list(session_dir.glob("*.pdf")))
        
        logger.info(f"[UPLOAD] File uploaded: {file.filename} ({file_size / (1024 * 1024):.2f}MB) to session {session_id}, total files: {file_count}")
        
        return {
            "success": True,
            "file_name": file.filename or "lecture.pdf",
            "session_id": session_id,
            "file_count": file_count,
            "file_size": file_size
        }
    except HTTPException:
        raise
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"Failed to upload file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@app.post("/generate")
async def generate_exam(
    request: Request,
    config: GenerateExamRequest = Body(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    生成考试PDF（支持匿名用户和认证用户）
    - 认证用户：检查用量限制
    - 匿名用户：每天只能使用1次（通过设备指纹识别）
    
    参数（JSON）：
    - session_id: 上传session ID
    - file_name: 可选，显示用的文件名（如果提供，将覆盖内部命名逻辑）
    - mcq_count: 选择题数量（默认10）
    - short_answer_count: 简答题数量（默认3）
    - long_question_count: 论述题数量（默认1）
    - difficulty: 难度等级（easy/medium/hard，默认medium）
    - special_requests: 用户特殊要求（可选，字符串）
    """
    try:
        session_id = config.session_id
        mcq_count = config.mcq_count
        short_answer_count = config.short_answer_count
        long_question_count = config.long_question_count
        difficulty = config.difficulty
        special_requests = config.special_requests
        
        # 记录接收到的参数（用于调试）
        logger.info(f"Received exam config: Session={session_id}, FileName={config.file_name if config.file_name else 'None (will use internal naming)'}, MCQ={mcq_count}, SAQ={short_answer_count}, LQ={long_question_count}, Difficulty={difficulty}, SpecialRequests={special_requests[:50] if special_requests else 'None'}...")
        
        # 验证session_id格式
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session_id format")
        
        # 从session目录读取文件
        session_dir = UPLOAD_SESSIONS_DIR / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found. Please upload files first.")
        
        # 获取所有PDF文件（按文件名排序）
        pdf_files = sorted(session_dir.glob("*.pdf"))
        if not pdf_files or len(pdf_files) == 0:
            raise HTTPException(status_code=400, detail="No PDF files found in session. Please upload at least one PDF file.")
        
        if len(pdf_files) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 PDF files allowed.")
        
        logger.info(f"Found {len(pdf_files)} PDF file(s) in session {session_id}")
        
        # 验证参数
        if mcq_count < 0 or mcq_count > 50:
            raise HTTPException(status_code=400, detail="mcq_count must be between 0 and 50")
        if short_answer_count < 0 or short_answer_count > 20:
            raise HTTPException(status_code=400, detail="short_answer_count must be between 0 and 20")
        if long_question_count < 0 or long_question_count > 10:
            raise HTTPException(status_code=400, detail="long_question_count must be between 0 and 10")
        # 支持 "normal" 映射到 "medium"（前端兼容）
        if difficulty == "normal":
            difficulty = "medium"
        if difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="difficulty must be one of: easy, medium, hard")

        # 判断是认证用户还是匿名用户
        if current_user:
            # 认证用户：检查用量限制
            user_id = current_user["id"]
            user_type = "authenticated"
            
            # 检查正常用量限制（已移除注册奖励逻辑）
            check_usage_limit(user_id)
            
            logger.info(f"Authenticated user {user_id} requesting job")
        else:
            # 匿名用户：获取或创建 anon_id（从Cookie）
            from fastapi import Response
            response = Response()
            anon_id = get_or_create_anon_id(request, response)
            
            # 检查匿名用户是否超过限制
            if not check_anon_usage_limit(anon_id):
                raise HTTPException(
                    status_code=429, 
                    detail="You've used your free daily limit. Register to unlock more features (difficulty adjustment, multiple PDFs, more exams)."
                )
            
            user_id = None  # 匿名用户
            user_type = "anonymous"
            logger.info(f"Anonymous user (anon_id: {anon_id[:8]}...) requesting job")

        job_id = str(uuid.uuid4())
        # 处理文件名：优先使用前端传递的 file_name，否则使用内部命名逻辑
        if config.file_name:
            # 使用前端传递的文件名
            file_name = config.file_name
            logger.info(f"[GENERATE] Using frontend-provided file_name: {file_name}")
        else:
            # 回退到内部命名逻辑
            if len(pdf_files) == 1:
                file_name = pdf_files[0].name
            else:
                file_name = f"{pdf_files[0].name} (+{len(pdf_files) - 1} more)"
            logger.info(f"[GENERATE] Using internal file_name: {file_name}")
        created_at = datetime.utcnow().isoformat()
        
        logger.info(f"[GENERATE] Starting job {job_id} for {user_type} user {user_id}, files: {len(pdf_files)} PDF(s) from session {session_id}")
        
        # 保存到数据库（商用级：移除内存状态，完全依赖数据库）
        try:
            with get_db() as conn:
                cur = conn.cursor()
                # 无论认证用户还是匿名用户，都保存设备指纹，用于备用验证
                device_fingerprint = get_device_fingerprint(request)
                logger.info(f"[GENERATE] Device fingerprint for job {job_id}: {device_fingerprint[:16]}...")
                if current_user:
                    # 认证用户：同时保存user_id和设备指纹
                    cur.execute(
                        """
                        INSERT INTO jobs (id, user_id, device_fingerprint, file_name, status, created_at, updated_at, mcq_count, short_answer_count, long_question_count, difficulty, special_requests)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (job_id, user_id, device_fingerprint, file_name, "queued", created_at, created_at, mcq_count, short_answer_count, long_question_count, difficulty, special_requests),
                    )
                    logger.info(f"[GENERATE] Job {job_id} saved to DB: user_id={user_id}, device_fp={device_fingerprint[:16]}...")
                    
                    # 验证job是否正确保存
                    cur.execute("SELECT id, user_id, file_name FROM jobs WHERE id = ?", (job_id,))
                    verify_row = cur.fetchone()
                    if verify_row:
                        logger.info(f"[GENERATE] Verified: Job {job_id} exists in DB with user_id={verify_row['user_id']}, file_name={verify_row['file_name']} (expected: {file_name})")
                        if verify_row['file_name'] != file_name:
                            logger.error(f"[GENERATE] WARNING: file_name mismatch! Expected '{file_name}', but DB has '{verify_row['file_name']}'")
                    else:
                        logger.error(f"[GENERATE] CRITICAL: Job {job_id} was not found in DB immediately after INSERT!")
                else:
                    # 匿名用户：记录设备指纹，user_id为NULL
                    cur.execute(
                        """
                        INSERT INTO jobs (id, user_id, device_fingerprint, file_name, status, created_at, updated_at, mcq_count, short_answer_count, long_question_count, difficulty, special_requests)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (job_id, None, device_fingerprint, file_name, "queued", created_at, created_at, mcq_count, short_answer_count, long_question_count, difficulty, special_requests),
                    )
                    logger.info(f"[GENERATE] Job {job_id} saved to DB: user_id=NULL, device_fp={device_fingerprint[:16]}..., file_name={file_name}")
                    
                    # 验证job是否正确保存（匿名用户）
                    cur.execute("SELECT id, user_id, file_name FROM jobs WHERE id = ?", (job_id,))
                    verify_row = cur.fetchone()
                    if verify_row:
                        logger.info(f"[GENERATE] Verified: Job {job_id} exists in DB with user_id={verify_row['user_id']}, file_name={verify_row['file_name']} (expected: {file_name})")
                        if verify_row['file_name'] != file_name:
                            logger.error(f"[GENERATE] WARNING: file_name mismatch! Expected '{file_name}', but DB has '{verify_row['file_name']}'")
                    else:
                        logger.error(f"[GENERATE] CRITICAL: Job {job_id} was not found in DB immediately after INSERT!")
            logger.info(f"[GENERATE] Job {job_id} created in database successfully")
            
            # 记录匿名使用（在数据库事务外，避免锁定）
            if not current_user:
                try:
                    record_anon_usage(anon_id)
                except Exception as e:
                    # 记录失败不影响主流程，只记录日志
                    logger.warning(f"Failed to record anon usage: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to create job in database: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to create job")
        
        # 使用 BUILD_ROOT 保持一致性
        job_dir = BUILD_ROOT / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create job directory: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to create job directory")

        # 从session目录复制文件到job目录（添加大小检查和错误处理，支持多个文件）
        lecture_paths = []
        total_size = 0
        try:
            for idx, source_file in enumerate(pdf_files):
                # 单个文件：lecture.pdf，多个文件：lecture_0.pdf, lecture_1.pdf, ...
                if len(pdf_files) == 1:
                    save_path = job_dir / "lecture.pdf"
                else:
                    save_path = job_dir / f"lecture_{idx}.pdf"
                
                # 检查文件大小
                file_size = source_file.stat().st_size
                total_size += file_size
                
                if total_size > MAX_FILE_SIZE * len(pdf_files):  # 总大小限制：单个文件限制 × 文件数
                    raise HTTPException(
                        status_code=413,
                        detail=f"Total file size too large. Maximum total size is {MAX_FILE_SIZE * len(pdf_files) / (1024 * 1024):.0f}MB for {len(pdf_files)} file(s)"
                    )
                
                # 复制文件
                shutil.copy2(source_file, save_path)
                lecture_paths.append(save_path)
                logger.info(f"File {idx + 1}/{len(pdf_files)} copied: {source_file.name}, size: {file_size / (1024 * 1024):.2f}MB")
            
            logger.info(f"All {len(pdf_files)} file(s) copied, total size: {total_size / (1024 * 1024):.2f}MB")
            
            # 清理session目录（文件已复制到job目录）
            try:
                shutil.rmtree(session_dir)
                logger.info(f"Session directory {session_id} cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup session directory {session_id}: {e}", exc_info=True)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to save file: {e}", exc_info=True)
            # 清理：删除job目录和数据库记录
            try:
                if job_dir.exists():
                    shutil.rmtree(job_dir)
                with get_db() as conn:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        # 用量计数 +1（仅认证用户）
        if current_user:
            try:
                upsert_usage(current_user["id"])
            except Exception as e:
                logger.warning(f"Failed to update usage count: {e}", exc_info=True)
                # 不影响主流程，继续执行

        # 将任务加入队列（阶段1改进：使用队列和并发控制）
        # 传递exam配置参数和文件路径列表
        exam_config = {
            "mcq_count": mcq_count,
            "short_answer_count": short_answer_count,
            "long_question_count": long_question_count,
            "difficulty": difficulty,
            "special_requests": special_requests
        }
        try:
            # 传递文件路径列表（支持多个PDF）
            job_queue.put((job_id, lecture_paths, exam_config))
            update_queue_stats("enqueue")
            queue_size = queue_stats["current_queue_size"]
            processing = queue_stats["current_processing"]
            logger.info(
                f"Job {job_id} queued. Queue size: {queue_size}, "
                f"Processing: {processing}/{MAX_CONCURRENT_JOBS}"
            )
        except Exception as e:
            logger.error(f"Failed to queue job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to queue job for processing")

        # 如果是匿名用户，需要返回包含Cookie的响应
        if not current_user:
            from fastapi.responses import JSONResponse
            json_response = JSONResponse(content={"job_id": job_id, "status": "queued", "message": "Job queued successfully"})
            # 设置anon_id Cookie（如果需要）
            from fastapi import Response
            temp_response = Response()
            anon_id = get_or_create_anon_id(request, temp_response)
            if hasattr(temp_response, 'headers'):
                for header, value in temp_response.headers.items():
                    if header.lower() == "set-cookie":
                        json_response.headers.append(header, value)
            return json_response
        
        return {"job_id": job_id, "status": "queued", "message": "Job queued successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_exam: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/status/{job_id}")
async def job_status(
    request: Request,
    job_id: str, 
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    查询 job 状态（支持匿名用户和认证用户）
    """
    # 调试日志
    user_info = f"user_id={current_user['id']}" if current_user else "anonymous"
    logger.info(f"[STATUS] Request for job {job_id}, {user_info}, IP={request.client.host if request.client else 'unknown'}")
    
    # 从数据库读取任务状态
    with get_db() as conn:
        cur = conn.cursor()
        if current_user:
            # 认证用户：先通过user_id查询
            cur.execute(
                "SELECT status, error, download_url, user_id, device_fingerprint, progress_info FROM jobs WHERE id = ? AND user_id = ?",
                (job_id, current_user["id"]),
            )
            row = cur.fetchone()
            if row:
                logger.info(f"[STATUS] Found job {job_id} by user_id={current_user['id']}, status={row['status']}")
            else:
                logger.warning(f"[STATUS] Job {job_id} not found by user_id={current_user['id']}, trying device_fingerprint")
            # 如果没找到，尝试通过设备指纹查询（可能是IP/User-Agent变化）
            if not row:
                device_fingerprint = get_device_fingerprint(request)
                cur.execute(
                    "SELECT status, error, download_url, user_id, progress_info FROM jobs WHERE id = ? AND device_fingerprint = ?",
                    (job_id, device_fingerprint),
                )
                row = cur.fetchone()
                if row:
                    logger.info(f"[STATUS] Found job {job_id} by device_fingerprint, user_id in DB={row['user_id']}, current_user_id={current_user['id']}")
                    # 如果通过设备指纹找到，验证user_id是否匹配或为NULL
                    if row:
                        # SQLite返回的row是Row对象，使用字典式访问
                        if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                            logger.warning(f"[STATUS] Device fingerprint match but user_id mismatch: DB={row['user_id']}, current={current_user['id']}")
                            row = None  # 设备指纹匹配但user_id不匹配，拒绝
                        elif row["user_id"] is None:
                            logger.info(f"[STATUS] Found anonymous job {job_id} for authenticated user {current_user['id']}, will update user_id")
        else:
            # 匿名用户：通过device_fingerprint查询
            device_fingerprint = get_device_fingerprint(request)
            cur.execute(
                "SELECT status, error, download_url, progress_info FROM jobs WHERE id = ? AND device_fingerprint = ? AND user_id IS NULL",
                (job_id, device_fingerprint),
            )
            row = cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="job not found")

    status = row["status"]
    result = {"status": status}
    
    # 如果有进度信息，添加到响应中
    # SQLite Row对象：如果字段为NULL，row["progress_info"]会返回None
    try:
        progress_info = row["progress_info"]
        if progress_info:
            progress_data = json.loads(progress_info)
            # 确保返回的格式符合前端要求：只包含必要的字段
            result["progress"] = {
                "stage": progress_data.get("stage", ""),
                "current": progress_data.get("current", 0),
                "total": progress_data.get("total", 0),
                "message": progress_data.get("message", "")
            }
    except (KeyError, json.JSONDecodeError, TypeError, AttributeError):
        # 如果字段不存在、为NULL或解析失败，忽略进度信息
        pass
    
    if status == "done":
        return result
    elif status == "failed":
        result["error"] = row["error"] or "Unknown error"
        return result
    else:
        return result
@app.get("/download/{job_id}")
async def download_exam(
    request: Request,
    job_id: str, 
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    根据 job_id 返回对应的 PDF（支持匿名用户和认证用户）
    路径约定：build_jobs/{job_id}/build/exam_filled.pdf
    """
    # 调试日志
    user_info = f"user_id={current_user['id']}" if current_user else "anonymous"
    logger.info(f"[DOWNLOAD] Request for job {job_id}, {user_info}, IP={request.client.host if request.client else 'unknown'}")
    
    # 从数据库验证任务存在且属于当前用户或匿名设备（商用级：移除内存状态）
    with get_db() as conn:
        cur = conn.cursor()
        if current_user:
            # 认证用户：先尝试通过user_id查询
            cur.execute(
                "SELECT status, download_url, user_id, device_fingerprint FROM jobs WHERE id = ? AND user_id = ?",
                (job_id, current_user["id"]),
            )
            row = cur.fetchone()
            if row:
                logger.info(f"[DOWNLOAD] Found job {job_id} by user_id={current_user['id']}, status={row['status']}")
            else:
                logger.warning(f"[DOWNLOAD] Job {job_id} not found by user_id={current_user['id']}, trying device_fingerprint")
            
            # 如果没找到，尝试通过设备指纹验证（可能是IP/User-Agent变化导致user_id查询失败）
            if not row:
                device_fingerprint = get_device_fingerprint(request)
                cur.execute(
                    "SELECT status, download_url, user_id, device_fingerprint FROM jobs WHERE id = ? AND device_fingerprint = ?",
                    (job_id, device_fingerprint),
                )
                row = cur.fetchone()
                if row:
                    logger.info(f"[DOWNLOAD] Found job {job_id} by device_fingerprint, user_id in DB={row['user_id']}, current_user_id={current_user['id']}")
                # 如果通过设备指纹找到，且user_id匹配或为NULL，允许访问
                if row:
                    if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                        # 设备指纹匹配但user_id不匹配，可能是设备被其他用户使用过
                        logger.warning(f"[DOWNLOAD] Device fingerprint match but user_id mismatch: DB={row['user_id']}, current={current_user['id']}")
                        raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
                    # 如果user_id为NULL，可能是匿名用户创建后注册的job，更新user_id
                    elif row["user_id"] is None:
                        logger.info(f"[DOWNLOAD] Found anonymous job {job_id} for authenticated user {current_user['id']}, updating user_id")
                        cur.execute(
                            "UPDATE jobs SET user_id = ? WHERE id = ? AND user_id IS NULL",
                            (current_user["id"], job_id)
                        )
                        conn.commit()
            
            # 如果还是没找到，尝试最宽松的查询（仅通过job_id）
            if not row:
                cur.execute(
                    "SELECT status, download_url, user_id, device_fingerprint FROM jobs WHERE id = ?",
                    (job_id,),
                )
                row = cur.fetchone()
                if row:
                    logger.info(f"[DOWNLOAD] Found job {job_id} by job_id only, user_id in DB={row['user_id']}, current_user_id={current_user['id']}")
                # 如果找到但user_id不是当前用户，且不是NULL（匿名用户），则拒绝
                if row and row["user_id"] is not None and row["user_id"] != current_user["id"]:
                    logger.warning(f"[DOWNLOAD] Job {job_id} belongs to different user: DB={row['user_id']}, current={current_user['id']}")
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
                # 如果user_id为NULL，尝试通过设备指纹验证
                elif row and row["user_id"] is None:
                    device_fingerprint = get_device_fingerprint(request)
                    db_fingerprint = row['device_fingerprint'] if row['device_fingerprint'] else None
                    logger.info(f"[DOWNLOAD] Job {job_id} has NULL user_id, comparing device_fingerprint: DB={db_fingerprint[:16] if db_fingerprint else 'None'}..., current={device_fingerprint[:16]}...")
                    if row["device_fingerprint"] != device_fingerprint:
                        logger.warning(f"[DOWNLOAD] Device fingerprint mismatch for job {job_id}")
                        raise HTTPException(status_code=403, detail="Access denied: job belongs to another device")
                    # 设备指纹匹配，更新user_id
                    logger.info(f"[DOWNLOAD] Found anonymous job {job_id} for authenticated user {current_user['id']}, updating user_id")
                    cur.execute(
                        "UPDATE jobs SET user_id = ? WHERE id = ? AND user_id IS NULL",
                        (current_user["id"], job_id)
                    )
                    conn.commit()
        else:
            # 匿名用户：通过device_fingerprint查询
            device_fingerprint = get_device_fingerprint(request)
            cur.execute(
                "SELECT status, download_url, user_id, device_fingerprint FROM jobs WHERE id = ? AND device_fingerprint = ? AND user_id IS NULL",
                (job_id, device_fingerprint),
            )
            row = cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    
    if row["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # 从文件系统读取PDF
    job_dir = BUILD_ROOT / job_id
    pdf_path = job_dir / "build" / "exam_filled.pdf"

    if not pdf_path.exists():
        logger.error(f"PDF file not found for job {job_id} at {pdf_path}")
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="exam_generated.pdf",
    )


@app.get("/preview/{job_id}")
async def preview_exam(
    request: Request,
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """
    获取PDF第一页预览图（带水印）
    支持匿名用户和认证用户
    """
    # 验证job是否存在且属于当前用户/设备
    with get_db() as conn:
        cur = conn.cursor()
        if current_user:
            # 认证用户：验证 job 属于该用户
            cur.execute(
                "SELECT id, user_id, status FROM jobs WHERE id = ?",
                (job_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            # 如果 job 有 user_id，必须匹配
            if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
        else:
            # 匿名用户：验证 job 属于该设备
            device_fingerprint = get_device_fingerprint(request)
            cur.execute(
                "SELECT id, user_id, status, device_fingerprint FROM jobs WHERE id = ?",
                (job_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            # 如果 job 有 user_id，匿名用户不能访问
            if row["user_id"] is not None:
                raise HTTPException(status_code=403, detail="Access denied: this job belongs to a registered user. Please log in.")
            # 验证设备指纹
            if row["device_fingerprint"] != device_fingerprint:
                raise HTTPException(status_code=403, detail="Access denied: job belongs to another device")
    
    if row["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # 从文件系统读取预览图
    job_dir = BUILD_ROOT / job_id
    preview_path = job_dir / "build" / "preview.png"
    pdf_path = job_dir / "build" / "exam_filled.pdf"
    
    # 如果预览图不存在，尝试从PDF生成
    if not preview_path.exists():
        if pdf_path.exists():
            # 检查预览库是否可用
            if not PDF_PREVIEW_AVAILABLE:
                logger.warning(f"Preview image not found for job {job_id} and PDF preview libraries not available")
                raise HTTPException(
                    status_code=503, 
                    detail="Preview image generation is not available. Please install PyMuPDF and Pillow libraries."
                )
            
            logger.info(f"Preview image not found for job {job_id}, attempting to generate from PDF")
            try:
                _generate_preview_image(job_id, pdf_path, job_dir)
                # 重新检查预览图是否生成成功
                if not preview_path.exists():
                    logger.error(f"Failed to generate preview image for job {job_id}")
                    raise HTTPException(status_code=500, detail="Failed to generate preview image")
            except HTTPException:
                # 重新抛出HTTP异常
                raise
            except Exception as e:
                logger.error(f"Error generating preview image for job {job_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to generate preview image: {str(e)}")
        else:
            logger.warning(f"Preview image and PDF not found for job {job_id}")
            raise HTTPException(status_code=404, detail="Preview image not found and PDF not available")
    
    return FileResponse(
        path=str(preview_path),
        media_type="image/png",
        filename="preview.png",
    )


@app.get("/download_answer/{job_id}")
async def download_answer(
    request: Request,
    job_id: str, 
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    根据 job_id 返回对应的答案 PDF
    路径约定：build_jobs/{job_id}/build/answer_filled.pdf
    """
    # 调试日志
    user_info = f"user_id={current_user['id']}" if current_user else "anonymous"
    logger.info(f"[DOWNLOAD_ANSWER] Request for job {job_id}, {user_info}, IP={request.client.host}")
    
    # 验证job是否存在且属于当前用户
    with get_db() as conn:
        cur = conn.cursor()
        if current_user:
            # 认证用户：通过user_id查询
            cur.execute(
                "SELECT status, download_url, user_id FROM jobs WHERE id = ? AND user_id = ?",
                (job_id, current_user["id"]),
            )
            row = cur.fetchone()
        else:
            # 匿名用户：通过device_fingerprint查询
            device_fingerprint = get_device_fingerprint(request)
            cur.execute(
                "SELECT status, download_url, user_id, device_fingerprint FROM jobs WHERE id = ? AND device_fingerprint = ? AND user_id IS NULL",
                (job_id, device_fingerprint),
            )
            row = cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    
    if row["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # 从文件系统读取答案PDF
    job_dir = BUILD_ROOT / job_id
    answer_pdf_path = job_dir / "build" / "answer_filled.pdf"
    
    if not answer_pdf_path.exists():
        # 检查exam_data.json是否存在，以及是否有answers字段
        exam_data_path = job_dir / "exam_data.json"
        detail_msg = "Answer PDF file not found"
        
        if exam_data_path.exists():
            try:
                import json
                with open(exam_data_path, "r", encoding="utf-8") as f:
                    exam_data = json.load(f)
                
                if "answers" not in exam_data:
                    detail_msg = "Answer key was not generated for this exam. This may be an older exam created before the answer feature was added."
                else:
                    detail_msg = "Answer PDF was not generated. The answer data exists but PDF generation may have failed."
            except Exception as e:
                logger.warning(f"Failed to read exam_data.json for job {job_id}: {e}")
                detail_msg = "Answer PDF file not found. Unable to check exam data."
        else:
            detail_msg = "Answer PDF file not found. Exam data file is missing."
        
        logger.error(f"Answer PDF file not found for job {job_id} at {answer_pdf_path}. Detail: {detail_msg}")
        raise HTTPException(status_code=404, detail=detail_msg)
    
    return FileResponse(
        path=str(answer_pdf_path),
        media_type="application/pdf",
        filename="answer_key.pdf",
    )


@app.post("/generate_answer/{job_id}")
async def generate_answer(
    request: Request,
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """
    开始生成答案（付款后调用）
    触发异步答案生成任务（支持匿名用户和认证用户）
    """
    # 验证job是否存在且属于当前用户/设备
    with get_db() as conn:
        cur = conn.cursor()
        if current_user:
            # 认证用户：验证 job 属于该用户
            cur.execute(
                "SELECT id, user_id, status, answer_status FROM jobs WHERE id = ?",
                (job_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            # 如果 job 有 user_id，必须匹配
            if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
            user_id = current_user["id"]
            user_info = f"user_id={user_id}"
        else:
            # 匿名用户：验证 job 属于该设备
            device_fingerprint = get_device_fingerprint(request)
            cur.execute(
                "SELECT id, user_id, status, answer_status, device_fingerprint FROM jobs WHERE id = ?",
                (job_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            # 如果 job 有 user_id，匿名用户不能访问
            if row["user_id"] is not None:
                raise HTTPException(status_code=403, detail="Access denied: this job belongs to a registered user. Please log in.")
            # 验证设备指纹
            if row["device_fingerprint"] != device_fingerprint:
                raise HTTPException(status_code=403, detail="Access denied: job belongs to another device")
            user_id = None
            user_info = "anonymous"
    
    if row["status"] != "done":
        raise HTTPException(status_code=400, detail="Exam generation not completed yet")
    
    # sqlite3.Row 对象使用字典式访问，如果字段不存在会返回 None
    answer_status = row["answer_status"] if row["answer_status"] else "pending"
    
    # 如果答案已经在生成中或已完成，直接返回状态
    if answer_status in ["generating", "done"]:
        return {
            "status": answer_status,
            "message": "Answer generation already started or completed" if answer_status == "generating" else "Answer already generated"
        }
    
    # 更新状态为生成中
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET answer_status = ?, updated_at = ? WHERE id = ?",
            ("generating", datetime.utcnow().isoformat(), job_id),
        )
        conn.commit()
    
    logger.info(f"[GENERATE_ANSWER] Starting answer generation for job {job_id}, {user_info}")
    
    # 在后台线程中生成答案（异步）
    def generate_answer_async():
        try:
            _generate_answer_for_job(job_id, user_id)
        except Exception as e:
            logger.error(f"Failed to generate answer for job {job_id}: {e}", exc_info=True)
            # 更新状态为失败
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE jobs SET answer_status = ?, updated_at = ? WHERE id = ?",
                    ("failed", datetime.utcnow().isoformat(), job_id),
                )
                conn.commit()
    
    # 启动后台线程
    answer_thread = threading.Thread(target=generate_answer_async, daemon=True)
    answer_thread.start()
    
    return {
        "status": "generating",
        "message": "Answer generation started"
    }


@app.get("/answer_status/{job_id}")
async def answer_status(
    request: Request,
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """
    轮询答案生成状态（支持匿名用户和认证用户）
    返回: { status: "pending" | "generating" | "done" | "failed", error?: string }
    """
    # 验证job是否存在且属于当前用户/设备
    with get_db() as conn:
        cur = conn.cursor()
        if current_user:
            # 认证用户：验证 job 属于该用户
            cur.execute(
                "SELECT id, user_id, answer_status, error FROM jobs WHERE id = ?",
                (job_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            # 如果 job 有 user_id，必须匹配
            if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
        else:
            # 匿名用户：验证 job 属于该设备
            device_fingerprint = get_device_fingerprint(request)
            cur.execute(
                "SELECT id, user_id, answer_status, error, device_fingerprint FROM jobs WHERE id = ?",
                (job_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            # 如果 job 有 user_id，匿名用户不能访问
            if row["user_id"] is not None:
                raise HTTPException(status_code=403, detail="Access denied: this job belongs to a registered user. Please log in.")
            # 验证设备指纹
            if row["device_fingerprint"] != device_fingerprint:
                raise HTTPException(status_code=403, detail="Access denied: job belongs to another device")
    
    # sqlite3.Row 对象使用字典式访问
    answer_status = row["answer_status"] if row["answer_status"] else "pending"
    error = row["error"] if row["error"] else None
    
    result = {
        "status": answer_status
    }
    
    if error and answer_status == "failed":
        result["error"] = error
    
    return result


def _generate_answer_for_job(job_id: str, user_id: Optional[int] = None) -> bool:
    """
    为指定的job生成答案（完整流程：生成答案数据 + 生成答案PDF）
    
    返回：
    - True: 成功生成答案
    - False: 生成失败（会更新数据库状态）
    """
    try:
        job_dir = BUILD_ROOT / job_id
        exam_data_path = job_dir / "exam_data.json"
        
        if not exam_data_path.exists():
            logger.error(f"exam_data.json not found for job {job_id}, cannot generate answer")
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE jobs SET answer_status = ?, error = ?, updated_at = ? WHERE id = ?",
                    ("failed", "Exam data not found", datetime.utcnow().isoformat(), job_id),
                )
                conn.commit()
            return False
        
        # 读取exam_data
        import json
        with open(exam_data_path, "r", encoding="utf-8") as f:
            exam_data = json.load(f)
        
        # 检查是否已有答案数据
        if "answers" in exam_data:
            logger.info(f"Answer data already exists for job {job_id}, generating PDF only")
        else:
            # 生成答案数据
            user_info = f"user {user_id}" if user_id else "anonymous user"
            logger.info(f"Generating answer key for job {job_id} ({user_info})")
            
            # 导入generate_answer_key函数
            import sys
            sys.path.insert(0, str(BASE_DIR))
            from generate_exam_data import generate_answer_key
            
            # 生成答案
            answer_key = generate_answer_key(exam_data)
            exam_data["answers"] = answer_key
            
            # 保存更新后的exam_data.json
            with open(exam_data_path, "w", encoding="utf-8") as f:
                json.dump(exam_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Answer key generated and saved for job {job_id}")
        
        # 生成答案PDF
        logger.info(f"Generating answer PDF for job {job_id}")
        _render_and_compile_answer_pdf(job_id, job_dir, exam_data_path)
        
        # 检查答案PDF是否成功生成
        answer_pdf_path = job_dir / "build" / "answer_filled.pdf"
        if not answer_pdf_path.exists():
            raise RuntimeError("Answer PDF was not generated")
        
        # 更新状态为完成
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE jobs SET answer_status = ?, updated_at = ? WHERE id = ?",
                ("done", datetime.utcnow().isoformat(), job_id),
            )
            conn.commit()
        
        logger.info(f"Answer generation completed successfully for job {job_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate answer for job {job_id}: {e}", exc_info=True)
        # 更新状态为失败
        with get_db() as conn:
            cur = conn.cursor()
            error_msg = str(e)[:500]  # 限制错误消息长度
            cur.execute(
                "UPDATE jobs SET answer_status = ?, error = ?, updated_at = ? WHERE id = ?",
                ("failed", error_msg, datetime.utcnow().isoformat(), job_id),
            )
            conn.commit()
        return False