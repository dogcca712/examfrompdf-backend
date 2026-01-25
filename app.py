from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any
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
DB_PATH = BASE_DIR / "data.db"

JWT_SECRET = os.environ.get("JWT_SECRET", "change_me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_HOURS = 24
security = HTTPBearer(auto_error=False)
# Token黑名单改用数据库表（见init_db中的revoked_tokens表）

# Stripe 配置
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
stripe.api_key = STRIPE_API_KEY if STRIPE_API_KEY else None

PRICE_IDS = {
    "starter": os.environ.get("STRIPE_PRICE_STARTER", ""),
    "pro": os.environ.get("STRIPE_PRICE_PRO", ""),
}

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
    """数据库连接上下文管理器，确保连接正确关闭"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
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
                job_id, lecture_path, exam_config = item
            else:
                job_id, lecture_path = item[:2]
                exam_config = None
            
            # 获取信号量（如果已达到最大并发数，会阻塞等待）
            job_semaphore.acquire()
            update_queue_stats("dequeue")
            update_queue_stats("start_processing")
            
            try:
                logger.info(f"Worker thread: Starting job {job_id}")
                run_job(job_id, lecture_path, exam_config)
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
    
    # 关联匿名使用记录（如果存在）
    anon_id = request.cookies.get("anon_id")
    if anon_id:
        association_result = associate_anon_usage_to_user(anon_id, user_id)
        logger.info(f"Associated anon_id {anon_id[:8]}... to new user {user_id}, bonus: {association_result['bonus_count']}")
    else:
        # 如果没有 anon_id，仍然给注册奖励
        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO registration_bonus (user_id, bonus_count, used_count, created_at)
                VALUES (?, 3, 0, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    bonus_count = bonus_count + 3
                """,
                (user_id, now)
            )
        logger.info(f"Gave registration bonus to new user {user_id} (no anon_id)")
    
    logger.info(f"User registered: {email} (id: {user_id})")

    token = generate_jwt(user_id, email)
    plan = get_plan_for_user(user_id)
    
    result = {
        "access_token": token,
        "user": {
            "id": user_id,
            "email": email,
            "plan": plan
        },
        "bonus_count": 3  # 注册奖励次数
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
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] in ("checkout.session.completed",):
        session = event["data"]["object"]
        metadata = session.get("metadata", {})
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

    if event["type"] in ("customer.subscription.updated", "customer.subscription.deleted"):
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


# -------------------- 用量 API --------------------
@app.get("/usage/status")
async def usage_status(current_user=Depends(get_current_user)):
    plan = get_plan_for_user(current_user["id"])
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    daily_count, monthly_count = get_usage_counts(current_user["id"])
    
    daily_limit = limits.get("daily", 1)
    monthly_limit = limits.get("monthly", 30)
    
    # 计算是否可以生成
    if plan == "free":
        can_generate = daily_count < daily_limit
    else:
        can_generate = monthly_count < monthly_limit
    
    bonus_remaining = get_registration_bonus_remaining(current_user["id"])
    
    return {
        "plan": plan,
        "daily_used": daily_count,
        "daily_limit": daily_limit,
        "monthly_used": monthly_count,
        "monthly_limit": monthly_limit,
        "can_generate": can_generate or bonus_remaining > 0,  # 包括注册奖励
        "bonus_remaining": bonus_remaining,  # 注册奖励剩余次数
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
    with get_db() as conn:
        cur = conn.cursor()
        # 先获取总数（包括user_id匹配的，以及user_id为NULL但device_fingerprint匹配的）
        # 注意：这里只统计user_id匹配的，因为匿名用户的job不应该出现在认证用户的历史中
        cur.execute("SELECT COUNT(*) as total FROM jobs WHERE user_id = ?", (current_user["id"],))
        total = cur.fetchone()["total"]
        
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
    
    return {"jobs": jobs, "total": total}

def run_job(job_id: str, lecture_path: Path, exam_config: Optional[Dict[str, Any]] = None):
    """
    后台执行：PDF -> exam_data.json -> render_exam.py -> pdflatex -> PDF
    结果写入数据库（商用级改进：移除内存状态）
    
    exam_config: 包含 mcq_count, short_answer_count, long_question_count, difficulty
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

    # lecture_path 已经在 job_dir 中了（在 /generate 端点中保存的）
    # 如果 lecture_path 不在 job_dir 中，才需要复制
    job_lecture = job_dir / "lecture.pdf"
    if lecture_path != job_lecture:
        shutil.copy2(lecture_path, job_lecture)
    else:
        job_lecture = lecture_path  # 已经是正确位置了

    # 统一输出在 job_dir/build 下
    (job_dir / "build").mkdir(exist_ok=True)

    try:
        # 更新数据库状态（商用级：移除内存状态）
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?
                """,
                ("running", datetime.utcnow().isoformat(), job_id),
            )
        logger.info(f"Job {job_id} started processing")

        # 1) 生成 JSON - 直接输出到 job_dir，避免并发冲突
        job_exam_data_json = job_dir / "exam_data.json"
        env = os.environ.copy()
        # 传递exam配置参数到generate_exam_data.py
        env["EXAMGEN_MCQ_COUNT"] = str(exam_config["mcq_count"])
        env["EXAMGEN_SHORT_ANSWER_COUNT"] = str(exam_config["short_answer_count"])
        env["EXAMGEN_LONG_QUESTION_COUNT"] = str(exam_config["long_question_count"])
        env["EXAMGEN_DIFFICULTY"] = exam_config["difficulty"]
        if exam_config.get("special_requests"):
            env["EXAMGEN_SPECIAL_REQUESTS"] = exam_config["special_requests"]
        logger.info(f"Passing exam config to generate_exam_data.py: MCQ={exam_config['mcq_count']}, SAQ={exam_config['short_answer_count']}, LQ={exam_config['long_question_count']}, Difficulty={exam_config['difficulty']}, SpecialRequests={exam_config.get('special_requests', 'None')[:50] if exam_config.get('special_requests') else 'None'}...")
        subprocess.run(
            [sys.executable, str(BASE_DIR / "generate_exam_data.py"), str(job_lecture), str(job_exam_data_json)],
            cwd=str(BASE_DIR),
            check=True,
            env=env,
        )

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

        # 更新数据库（商用级：移除内存状态）
        download_url = f"/download/{job_id}"
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs SET status = ?, download_url = ?, updated_at = ? WHERE id = ?
                """,
                ("done", download_url, datetime.utcnow().isoformat(), job_id),
            )
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        error_msg = str(e)
        # 更新数据库（商用级：移除内存状态）
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?
                """,
                ("failed", error_msg, datetime.utcnow().isoformat(), job_id),
            )
        logger.error(f"Job {job_id} failed: {error_msg}", exc_info=True)


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

@app.post("/generate")
async def generate_exam(
    request: Request,
    lecture_pdf: UploadFile = File(...),
    mcq_count: int = Form(10),
    short_answer_count: int = Form(3),
    long_question_count: int = Form(1),
    difficulty: str = Form("medium"),
    special_requests: Optional[str] = Form(None),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    生成考试PDF（支持匿名用户和认证用户）
    - 认证用户：检查用量限制
    - 匿名用户：每天只能使用1次（通过设备指纹识别）
    
    参数：
    - mcq_count: 选择题数量（默认10）
    - short_answer_count: 简答题数量（默认3）
    - long_question_count: 论述题数量（默认1）
    - difficulty: 难度等级（easy/medium/hard，默认medium）
    - special_requests: 用户特殊要求（可选，字符串）
    """
    try:
        # 记录接收到的参数（用于调试）
        logger.info(f"Received exam config: MCQ={mcq_count}, SAQ={short_answer_count}, LQ={long_question_count}, Difficulty={difficulty}, SpecialRequests={special_requests[:50] if special_requests else 'None'}...")
        
        # 验证文件类型
        if lecture_pdf.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")
        
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
            # 认证用户：检查用量限制（包括注册奖励）
            user_id = current_user["id"]
            user_type = "authenticated"
            
            # 检查是否有注册奖励可用
            bonus_remaining = get_registration_bonus_remaining(user_id)
            if bonus_remaining > 0:
                # 使用注册奖励
                use_registration_bonus(user_id)
                logger.info(f"Authenticated user {user_id} using registration bonus (remaining: {bonus_remaining - 1})")
            else:
                # 检查正常用量限制
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
        file_name = lecture_pdf.filename or "lecture.pdf"
        created_at = datetime.utcnow().isoformat()
        
        logger.info(f"[GENERATE] Starting job {job_id} for {user_type} user {user_id}, file: {file_name}")
        
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
                else:
                    # 匿名用户：记录设备指纹，user_id为NULL
                    cur.execute(
                        """
                        INSERT INTO jobs (id, user_id, device_fingerprint, file_name, status, created_at, updated_at, mcq_count, short_answer_count, long_question_count, difficulty, special_requests)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (job_id, None, device_fingerprint, file_name, "queued", created_at, created_at, mcq_count, short_answer_count, long_question_count, difficulty, special_requests),
                    )
                    logger.info(f"[GENERATE] Job {job_id} saved to DB: user_id=NULL, device_fp={device_fingerprint[:16]}...")
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

        # 保存文件（添加大小检查和错误处理）
    lecture_path = job_dir / "lecture.pdf"
        file_size = 0
        try:
            with lecture_path.open("wb") as f:
                # 分块读取，避免内存问题，同时检查大小
                # 使用 read() 方法（同步，但在异步上下文中可以接受）
                chunk_size = 8192  # 8KB chunks
                while True:
                    chunk = lecture_pdf.file.read(chunk_size)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > MAX_FILE_SIZE:
                        lecture_path.unlink(missing_ok=True)  # 删除部分文件
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.0f}MB"
                        )
                    f.write(chunk)
            
            logger.info(f"File saved: {file_name}, size: {file_size / (1024 * 1024):.2f}MB")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to save file: {e}", exc_info=True)
            # 清理：删除job目录和数据库记录
            try:
                import shutil
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
        # 传递exam配置参数
        exam_config = {
            "mcq_count": mcq_count,
            "short_answer_count": short_answer_count,
            "long_question_count": long_question_count,
            "difficulty": difficulty,
            "special_requests": special_requests
        }
        try:
            job_queue.put((job_id, lecture_path, exam_config))
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
            # 复制Cookie设置
            for header, value in response.headers.items():
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
                "SELECT status, error, download_url, user_id, device_fingerprint FROM jobs WHERE id = ? AND user_id = ?",
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
                    "SELECT status, error, download_url, user_id FROM jobs WHERE id = ? AND device_fingerprint = ?",
                    (job_id, device_fingerprint),
                )
                row = cur.fetchone()
                if row:
                    logger.info(f"[STATUS] Found job {job_id} by device_fingerprint, user_id in DB={row.get('user_id')}, current_user_id={current_user['id']}")
                # 如果通过设备指纹找到，验证user_id是否匹配或为NULL
                if row:
                    # SQLite返回的row是字典，可以直接访问
                    if row.get("user_id") is not None and row.get("user_id") != current_user["id"]:
                        logger.warning(f"[STATUS] Device fingerprint match but user_id mismatch: DB={row.get('user_id')}, current={current_user['id']}")
                        row = None  # 设备指纹匹配但user_id不匹配，拒绝
                    elif row.get("user_id") is None:
                        logger.info(f"[STATUS] Found anonymous job {job_id} for authenticated user {current_user['id']}, will update user_id")
        else:
            # 匿名用户：通过device_fingerprint查询
            device_fingerprint = get_device_fingerprint(request)
            cur.execute(
                "SELECT status, error, download_url FROM jobs WHERE id = ? AND device_fingerprint = ? AND user_id IS NULL",
                (job_id, device_fingerprint),
            )
            row = cur.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="job not found")

    status = row["status"]
    if status == "done":
        return {"status": "done"}
    elif status == "failed":
        return {
            "status": "failed",
            "error": row["error"] or "Unknown error"
        }
    else:
        return {"status": status}
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
                    logger.info(f"[DOWNLOAD] Found job {job_id} by device_fingerprint, user_id in DB={row.get('user_id')}, current_user_id={current_user['id']}")
                # 如果通过设备指纹找到，且user_id匹配或为NULL，允许访问
                if row:
                    if row["user_id"] is not None and row["user_id"] != current_user["id"]:
                        # 设备指纹匹配但user_id不匹配，可能是设备被其他用户使用过
                        logger.warning(f"[DOWNLOAD] Device fingerprint match but user_id mismatch: DB={row.get('user_id')}, current={current_user['id']}")
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
                    logger.info(f"[DOWNLOAD] Found job {job_id} by job_id only, user_id in DB={row.get('user_id')}, current_user_id={current_user['id']}")
                # 如果找到但user_id不是当前用户，且不是NULL（匿名用户），则拒绝
                if row and row["user_id"] is not None and row["user_id"] != current_user["id"]:
                    logger.warning(f"[DOWNLOAD] Job {job_id} belongs to different user: DB={row.get('user_id')}, current={current_user['id']}")
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
                # 如果user_id为NULL，尝试通过设备指纹验证
                elif row and row["user_id"] is None:
                    device_fingerprint = get_device_fingerprint(request)
                    logger.info(f"[DOWNLOAD] Job {job_id} has NULL user_id, comparing device_fingerprint: DB={row.get('device_fingerprint')[:16] if row.get('device_fingerprint') else 'None'}..., current={device_fingerprint[:16]}...")
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