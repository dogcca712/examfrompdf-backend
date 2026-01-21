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

BASE_DIR = Path(__file__).parent
BUILD_ROOT = BASE_DIR / "build_jobs"
BUILD_ROOT.mkdir(exist_ok=True)
BUILD_DIR = BASE_DIR / "build"
DB_PATH = BASE_DIR / "data.db"

JWT_SECRET = os.environ.get("JWT_SECRET", "change_me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_HOURS = 24
security = HTTPBearer(auto_error=False)
TOKEN_BLACKLIST = set()  # 简单的内存黑名单

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

# 内存 Job 状态（MVP 够用；以后可换 Redis / DB）
JOBS = {}  # job_id -> dict(status, error, pdf_path, created_at, user_id)


# -------------------- 数据库工具 --------------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
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
            stripe_subscription_id TEXT,
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
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()


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
        if jti in TOKEN_BLACKLIST:
            raise HTTPException(status_code=401, detail="Token revoked")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
    row = cur.fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def upsert_usage(user_id: int):
    today = date.today().isoformat()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO usage (user_id, date, count) VALUES (?, ?, 1)
        ON CONFLICT(user_id, date) DO UPDATE SET count = count + 1
        """,
        (user_id, today),
    )
    conn.commit()
    conn.close()


def get_usage_counts(user_id: int):
    today = date.today().isoformat()
    month_prefix = today[:7]  # YYYY-MM
    conn = get_db()
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
    conn.close()
    return daily_count, monthly_count


def get_active_subscription(user_id: int) -> Optional[sqlite3.Row]:
    conn = get_db()
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
    row = cur.fetchone()
    conn.close()
    return row


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
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
        (email, password_hash, datetime.utcnow().isoformat()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()

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
    jti = current_user.get("token_jti")
    if jti:
        TOKEN_BLACKLIST.add(jti)
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
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO subscriptions (user_id, plan, stripe_customer_id, stripe_subscription_id, status, current_period_end)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, plan, customer_id, subscription_id, status, current_period_end),
    )
    conn.commit()
    conn.close()


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
    获取用户的历史任务列表
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
    conn = get_db()
    cur = conn.cursor()
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
    conn.close()
    
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
    
    return {"jobs": jobs, "total": len(jobs)}

def run_job(job_id: str, lecture_path: Path):
    """
    后台执行：PDF -> exam_data.json -> render_exam.py -> pdflatex -> PDF
    结果写入 JOBS[job_id]
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
        JOBS[job_id]["status"] = "running"
        
        # 更新数据库状态
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE jobs SET status = ? WHERE id = ?
            """,
            ("running", job_id),
        )
        conn.commit()
        conn.close()

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

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["pdf_path"] = str(pdf_path)
        
        # 更新数据库
        conn = get_db()
        cur = conn.cursor()
        download_url = f"/download/{job_id}"
        cur.execute(
            """
            UPDATE jobs SET status = ?, download_url = ? WHERE id = ?
            """,
            ("done", download_url, job_id),
        )
        conn.commit()
        conn.close()
        
        # 更新数据库
        conn = get_db()
        cur = conn.cursor()
        download_url = f"/download/{job_id}"
        cur.execute(
            """
            UPDATE jobs SET status = ?, download_url = ? WHERE id = ?
            """,
            ("done", download_url, job_id),
        )
        conn.commit()
        conn.close()

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        
        # 更新数据库
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE jobs SET status = ?, error = ? WHERE id = ?
            """,
            ("failed", str(e), job_id),
        )
        conn.commit()
        conn.close()


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


@app.post("/generate")
async def generate_exam(lecture_pdf: UploadFile = File(...), current_user=Depends(get_current_user)):
    if lecture_pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    # 用量检查
    check_usage_limit(current_user["id"])

    job_id = str(uuid.uuid4())
    
    # 初始化 job 状态
    JOBS[job_id] = {
        "status": "queued",
        "error": None,
        "pdf_path": None,
        "created_at": time.time(),
        "user_id": current_user["id"],
    }
    
    # 使用 BUILD_ROOT 保持一致性
    job_dir = BUILD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    lecture_path = job_dir / "lecture.pdf"
    with lecture_path.open("wb") as f:
        shutil.copyfileobj(lecture_pdf.file, f)

    # 用量计数 +1
    upsert_usage(current_user["id"])

    # 使用后台线程执行任务，避免阻塞
    thread = threading.Thread(target=run_job, args=(job_id, lecture_path))
    thread.daemon = True
    thread.start()

    return JSONResponse({"job_id": job_id})
@app.get("/status/{job_id}")
async def job_status(job_id: str, current_user=Depends(get_current_user)):
    """
    查询 job 状态
    """
    # 先检查内存中的状态
    if job_id in JOBS:
        job = JOBS[job_id]
        if job.get("user_id") and job["user_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        status = job.get("status", "unknown")
        
        # 如果状态是 done，返回 PDF 路径
        if status == "done":
            return {
                "status": "done"
            }
        elif status == "failed":
            return {
                "status": "failed",
                "error": job.get("error", "Unknown error")
            }
        else:
            return {"status": status}
    
    # 如果内存中没有，检查文件系统（向后兼容）
    job_dir = BUILD_ROOT / job_id
    pdf_path = job_dir / "build" / "exam_filled.pdf"

    if pdf_path.exists():
        return {
            "status": "done"
        }
    elif job_dir.exists():
        return {"status": "running"}
    else:
        raise HTTPException(status_code=404, detail="job not found")
@app.get("/download/{job_id}")
async def download_exam(job_id: str, current_user=Depends(get_current_user)):
    """
    根据 job_id 返回对应的 PDF
    路径约定：build_jobs/{job_id}/build/exam_filled.pdf
    """
    # 优先从内存状态获取路径
    if job_id in JOBS:
        pdf_path_str = JOBS[job_id].get("pdf_path")
        if JOBS[job_id].get("user_id") and JOBS[job_id]["user_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        if pdf_path_str:
            pdf_path = Path(pdf_path_str)
            if pdf_path.exists():
                return FileResponse(
                    path=str(pdf_path),
                    media_type="application/pdf",
                    filename="exam_generated.pdf",
                )
    
    # 向后兼容：从文件系统查找
    job_dir = BUILD_ROOT / job_id
    pdf_path = job_dir / "build" / "exam_filled.pdf"

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="job not found or PDF not ready")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="exam_generated.pdf",
    )