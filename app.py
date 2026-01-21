from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
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
        # 匿名用户追踪表（用于限制未登录用户）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS guest_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_fingerprint TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                date TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(device_fingerprint, date)
            );
            """
        )
        # 匿名用户到注册用户的关联表（用于注册时关联使用记录）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS guest_to_user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_fingerprint TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                associated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                UNIQUE(device_fingerprint, user_id)
            );
            """
        )
        # 创建索引提升查询性能
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_guest_usage_fingerprint ON guest_usage(device_fingerprint)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_guest_usage_date ON guest_usage(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_guest_to_user_fingerprint ON guest_to_user(device_fingerprint)")


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
                # 1. 创建新表（允许 user_id 为 NULL）
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
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )
                """)
                
                # 2. 复制数据（保留所有现有数据）
                cur.execute("""
                    INSERT INTO jobs_new 
                    (id, user_id, device_fingerprint, file_name, status, created_at, download_url, error, updated_at)
                    SELECT 
                        id, 
                        user_id,
                        device_fingerprint,
                        file_name, 
                        status, 
                        created_at, 
                        download_url, 
                        error,
                        COALESCE(updated_at, created_at)
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
            job_id, lecture_path = job_queue.get()
            
            # 获取信号量（如果已达到最大并发数，会阻塞等待）
            job_semaphore.acquire()
            update_queue_stats("dequeue")
            update_queue_stats("start_processing")
            
            try:
                logger.info(f"Worker thread: Starting job {job_id}")
                run_job(job_id, lecture_path)
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
    """生成设备指纹：IP + User-Agent"""
    ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    # 使用hash确保隐私，同时保持一致性
    fingerprint_str = f"{ip}:{user_agent}"
    fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]  # 32字符足够
    return fingerprint


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


def associate_guest_usage_to_user(device_fingerprint: str, user_id: int):
    """用户注册时，关联匿名使用记录到用户账户"""
    today = date.today().isoformat()
    now = datetime.utcnow().isoformat()
    
    with get_db() as conn:
        cur = conn.cursor()
        # 检查今天是否有匿名使用记录
        cur.execute(
            """
            SELECT count FROM guest_usage 
            WHERE device_fingerprint = ? AND date = ?
            """,
            (device_fingerprint, today)
        )
        row = cur.fetchone()
        
        if row and row[0] > 0:
            # 有匿名使用记录，关联到用户并消耗免费额度
            cur.execute(
                """
                INSERT OR IGNORE INTO guest_to_user (device_fingerprint, user_id, associated_at)
                VALUES (?, ?, ?)
                """,
                (device_fingerprint, user_id, now)
            )
            # 将匿名使用记录转移到用户使用记录
            cur.execute(
                """
                INSERT INTO usage (user_id, date, count)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, date) DO UPDATE SET
                    count = count + ?
                """,
                (user_id, today, row[0], row[0])
            )
            logger.info(f"Associated guest usage ({row[0]} times) to user {user_id}")
            return row[0]
    return 0


# -------------------- 认证 API --------------------
@app.post("/auth/register")
async def register(request: Request, payload: Dict[str, str]):
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
    device_fingerprint = get_device_fingerprint(request)
    guest_usage_count = associate_guest_usage_to_user(device_fingerprint, user_id)
    if guest_usage_count > 0:
        logger.info(f"Associated {guest_usage_count} guest usage(s) to new user {user_id}")
    
    logger.info(f"User registered: {email} (id: {user_id})")

    token = generate_jwt(user_id, email)
    plan = get_plan_for_user(user_id)
    return {
        "access_token": token,
        "user": {
            "id": user_id,
            "email": email,
            "plan": plan
        }
    }


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
    
    return {
        "plan": plan,
        "daily_used": daily_count,
        "daily_limit": daily_limit,
        "monthly_used": monthly_count,
        "monthly_limit": monthly_limit,
        "can_generate": can_generate,
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

def run_job(job_id: str, lecture_path: Path):
    """
    后台执行：PDF -> exam_data.json -> render_exam.py -> pdflatex -> PDF
    结果写入数据库（商用级改进：移除内存状态）
    """
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
        subprocess.run(
            [sys.executable, str(BASE_DIR / "generate_exam_data.py"), str(job_lecture), str(job_exam_data_json)],
            cwd=str(BASE_DIR),
            check=True,
            env=os.environ.copy(),
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
        
        subprocess.run(
            [compiler, "-interaction=nonstopmode", "-output-directory", str(job_dir / "build"), str(tex_path)],
            check=True,
        )

        pdf_path = job_dir / "build" / "exam_filled.pdf"
        if not pdf_path.exists():
            raise RuntimeError("PDF was not generated.")

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
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    生成考试PDF（支持匿名用户和认证用户）
    - 认证用户：检查用量限制
    - 匿名用户：每天只能使用1次（通过设备指纹识别）
    """
    try:
        # 验证文件类型
        if lecture_pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        # 判断是认证用户还是匿名用户
        if current_user:
            # 认证用户：检查用量限制
            check_usage_limit(current_user["id"])
            user_id = current_user["id"]
            user_type = "authenticated"
            logger.info(f"Authenticated user {user_id} requesting job")
        else:
            # 匿名用户：检查设备指纹限制
            device_fingerprint = get_device_fingerprint(request)
            ip_address = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            
            if not check_guest_usage_limit(device_fingerprint):
                raise HTTPException(
                    status_code=429, 
                    detail="Anonymous users can only generate one exam per day. Please register for more."
                )
            
            # 使用特殊用户ID 0 表示匿名用户（或创建临时用户记录）
            # 但为了数据库完整性，我们使用一个特殊的guest用户ID
            # 或者直接在jobs表中使用NULL，但需要修改表结构
            # 暂时使用0作为guest user_id，但需要确保数据库允许
            user_id = 0  # 匿名用户
            user_type = "guest"
            logger.info(f"Guest user (fingerprint: {device_fingerprint[:8]}...) requesting job")
            
            # 记录匿名使用（在创建job之前，如果失败不会记录）
            # 注意：这里先不记录，等job创建成功后再记录

    job_id = str(uuid.uuid4())
        file_name = lecture_pdf.filename or "lecture.pdf"
        created_at = datetime.utcnow().isoformat()
        
        logger.info(f"Starting job {job_id} for {user_type} user {user_id}, file: {file_name}")
        
        # 保存到数据库（商用级：移除内存状态，完全依赖数据库）
        try:
            with get_db() as conn:
                cur = conn.cursor()
                if current_user:
                    # 认证用户
                    cur.execute(
                        """
                        INSERT INTO jobs (id, user_id, file_name, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (job_id, user_id, file_name, "queued", created_at, created_at),
                    )
                else:
                    # 匿名用户：记录设备指纹
                    device_fingerprint = get_device_fingerprint(request)
                    cur.execute(
                        """
                        INSERT INTO jobs (id, user_id, device_fingerprint, file_name, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (job_id, None, device_fingerprint, file_name, "queued", created_at, created_at),
                    )
            logger.info(f"Job {job_id} created in database")
            
            # 记录匿名使用（在数据库事务外，避免锁定）
            if not current_user:
                device_fingerprint = get_device_fingerprint(request)
                ip_address = request.client.host if request.client else "unknown"
                user_agent = request.headers.get("user-agent", "unknown")
                try:
                    record_guest_usage(device_fingerprint, ip_address, user_agent)
                except Exception as e:
                    # 记录失败不影响主流程，只记录日志
                    logger.warning(f"Failed to record guest usage: {e}", exc_info=True)
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
        try:
            job_queue.put((job_id, lecture_path))
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

    return JSONResponse({"job_id": job_id})
    
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
    # 从数据库读取任务状态
    with get_db() as conn:
        cur = conn.cursor()
        if current_user:
            # 认证用户：通过user_id查询
            cur.execute(
                "SELECT status, error, download_url FROM jobs WHERE id = ? AND user_id = ?",
                (job_id, current_user["id"]),
            )
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
            
            # 如果没找到，可能是匿名用户创建后注册的job（user_id为NULL）
            # 或者重建表前的job（user_id可能为0），尝试更宽松的查询
            if not row:
                cur.execute(
                    "SELECT status, download_url, user_id, device_fingerprint FROM jobs WHERE id = ?",
                    (job_id,),
                )
                row = cur.fetchone()
                # 如果找到但user_id不是当前用户，且不是NULL（匿名用户），则拒绝
                if row and row["user_id"] is not None and row["user_id"] != current_user["id"]:
                    raise HTTPException(status_code=403, detail="Access denied: job belongs to another user")
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