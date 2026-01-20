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
import uuid
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
    job_dir = BUILD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    lecture_path = job_dir / "lecture.pdf"
    with lecture_path.open("wb") as f:
        shutil.copyfileobj(lecture_pdf.file, f)

    # 运行生成 JSON（注意：这里要用 sys.executable，并且传 lecture_path）
    try:
        subprocess.run(
            [sys.executable, "generate_exam_data.py", str(lecture_path)],
            check=True,
            cwd=BASE_DIR,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"generate_exam_data.py failed: {e}")

    # 渲染 tex（你 render_exam.py 如果读固定 exam_data.json，也建议改成读 job_dir 内的）
    try:
        subprocess.run(
            [sys.executable, "render_exam.py"],
            cwd=BASE_DIR,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"render_exam.py failed: {e}")

    # 编译 PDF（输出到 job_dir）
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", str(job_dir),
             str(BASE_DIR / "build" / "exam_filled.tex")],
            cwd=BASE_DIR,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"pdflatex failed: {e}")

    pdf_path = job_dir / "exam_filled.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail="PDF was not generated.")

    return JSONResponse({"job_id": job_id})
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Very simple job status:
    - 如果该 job 的 exam_filled.pdf 已经生成 -> done
    - 如果 job 目录存在但 pdf 还没出来 -> running
    - 如果连目录都没有 -> queued（前端会继续轮询）
    """
    job_dir = BUILD_DIR / job_id
    pdf_path = job_dir / "exam_filled.pdf"

    if pdf_path.exists():
        return {"status": "done"}
    elif job_dir.exists():
        return {"status": "running"}
    else:
        return {"status": "queued"}

@app.get("/status/{job_id}")
async def job_status(job_id: str):
    job_dir = BUILD_DIR / job_id
    pdf_path = job_dir / "exam_filled.pdf"

    # job 不存在
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")

    # 已生成完成
    if pdf_path.exists():
        return {
            "status": "done"
        }

    # 还在处理中
    return {
        "status": "running"
    }
@app.get("/download/{job_id}")
async def download_exam(job_id: str):
    """
    根据 job_id 返回对应的 PDF
    路径约定：build/{job_id}/exam_filled.pdf
    """
    job_dir = BUILD_DIR / job_id
    pdf_path = job_dir / "exam_filled.pdf"

    if not pdf_path.exists():
        # 被前端看到就是 404 + "job not found"
        raise HTTPException(status_code=404, detail="job not found")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="exam_generated.pdf",
    )