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
                user_id INTEGER NOT NULL,
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
        # 创建索引提升查询性能
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")


init_db()


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


# -------------------- 认证 API --------------------
@app.post("/auth/register")
async def register(payload: Dict[str, str]):
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
        # 先获取总数
        cur.execute("SELECT COUNT(*) as total FROM jobs WHERE user_id = ?", (current_user["id"],))
        total = cur.fetchone()["total"]
        
        # 获取任务列表
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
    return {"status": "ok"}


# 文件大小限制：50MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.post("/generate")
async def generate_exam(lecture_pdf: UploadFile = File(...), current_user=Depends(get_current_user)):
    """
    生成考试PDF（商用级：添加文件大小检查和错误处理）
    """
    try:
        # 验证文件类型
    if lecture_pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        # 用量检查
        check_usage_limit(current_user["id"])

    job_id = str(uuid.uuid4())
        file_name = lecture_pdf.filename or "lecture.pdf"
        created_at = datetime.utcnow().isoformat()
        
        logger.info(f"Starting job {job_id} for user {current_user['id']}, file: {file_name}")
        
        # 保存到数据库（商用级：移除内存状态，完全依赖数据库）
        try:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO jobs (id, user_id, file_name, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, current_user["id"], file_name, "queued", created_at, created_at),
                )
            logger.info(f"Job {job_id} created in database")
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

        # 用量计数 +1
        try:
            upsert_usage(current_user["id"])
        except Exception as e:
            logger.warning(f"Failed to update usage count: {e}", exc_info=True)
            # 不影响主流程，继续执行

        # 使用后台线程执行任务，避免阻塞
        try:
            thread = threading.Thread(target=run_job, args=(job_id, lecture_path))
            thread.daemon = True
            thread.start()
            logger.info(f"Background job thread started for {job_id}")
        except Exception as e:
            logger.error(f"Failed to start background job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to start processing")

    return JSONResponse({"job_id": job_id})
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_exam: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.get("/status/{job_id}")
async def job_status(job_id: str, current_user=Depends(get_current_user)):
    """
    查询 job 状态（商用级：完全从数据库读取）
    """
    # 从数据库读取任务状态
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT status, error, download_url FROM jobs WHERE id = ? AND user_id = ?",
            (job_id, current_user["id"]),
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
async def download_exam(job_id: str, current_user=Depends(get_current_user)):
    """
    根据 job_id 返回对应的 PDF
    路径约定：build_jobs/{job_id}/build/exam_filled.pdf
    """
    # 从数据库验证任务存在且属于当前用户（商用级：移除内存状态）
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT status, download_url FROM jobs WHERE id = ? AND user_id = ?",
            (job_id, current_user["id"]),
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