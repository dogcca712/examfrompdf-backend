from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import subprocess
import os
import sys
import uuid
import threading
import time
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).parent
BUILD_ROOT = BASE_DIR / "build_jobs"
BUILD_ROOT.mkdir(exist_ok=True)
BUILD_DIR = BASE_DIR / "build"

app = FastAPI(title="ExamFromPDF MVP")

# CORS: 先放开，等你域名/前端稳定后再收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存 Job 状态（MVP 够用；以后可换 Redis / DB）
JOBS = {}  # job_id -> dict(status, error, pdf_path, created_at)

def run_job(job_id: str, lecture_path: Path):
    """
    后台执行：PDF -> exam_data.json -> render_exam.py -> pdflatex -> PDF
    结果写入 JOBS[job_id]
    """
    job_dir = BUILD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 把 lecture.pdf 固定放到 job_dir 里，避免并发互相覆盖
    job_lecture = job_dir / "lecture.pdf"
    shutil.copy2(lecture_path, job_lecture)

    # 统一输出在 job_dir/build 下
    (job_dir / "build").mkdir(exist_ok=True)

    try:
        JOBS[job_id]["status"] = "running"

        # 1) 生成 JSON
        subprocess.run(
            [sys.executable, str(BASE_DIR / "generate_exam_data.py"), str(job_lecture)],
            cwd=str(BASE_DIR),
            check=True,
            env=os.environ.copy(),
        )

        # 2) 渲染 tex（render_exam.py 读取 BASE_DIR/exam_data.json 并输出 build/exam_filled.tex）
        #    为了并发安全：我们把 exam_data.json / build 文件复制到 job_dir 下再编译
        #    这里用最简单的方式：生成后把文件拷到 job_dir，再在 job_dir 里运行 render+latex
        shutil.copy2(BASE_DIR / "exam_data.json", job_dir / "exam_data.json")

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

        # 3) 编译 PDF
        tex_path = job_dir / "build" / "exam_filled.tex"
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(job_dir / "build"), str(tex_path)],
            check=True,
        )

        pdf_path = job_dir / "build" / "exam_filled.pdf"
        if not pdf_path.exists():
            raise RuntimeError("PDF was not generated.")

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["pdf_path"] = str(pdf_path)

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


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
async def generate_exam(lecture_pdf: UploadFile = File(...)):
    if lecture_pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    job_id = str(uuid.uuid4())
    
    # 初始化 job 状态
    JOBS[job_id] = {
        "status": "queued",
        "error": None,
        "pdf_path": None,
        "created_at": time.time()
    }
    
    # 使用 BUILD_ROOT 保持一致性
    job_dir = BUILD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    lecture_path = job_dir / "lecture.pdf"
    with lecture_path.open("wb") as f:
        shutil.copyfileobj(lecture_pdf.file, f)

    # 使用后台线程执行任务，避免阻塞
    thread = threading.Thread(target=run_job, args=(job_id, lecture_path))
    thread.daemon = True
    thread.start()

    return JSONResponse({"job_id": job_id})
@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """
    查询 job 状态
    """
    # 先检查内存中的状态
    if job_id in JOBS:
        job = JOBS[job_id]
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
async def download_exam(job_id: str):
    """
    根据 job_id 返回对应的 PDF
    路径约定：build_jobs/{job_id}/build/exam_filled.pdf
    """
    # 优先从内存状态获取路径
    if job_id in JOBS:
        pdf_path_str = JOBS[job_id].get("pdf_path")
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