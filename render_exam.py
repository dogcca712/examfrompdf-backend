import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import os

BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = Path(os.environ.get("EXAMGEN_OUTPUT_DIR", str(BASE_DIR / "build")))
EXAM_DATA_PATH = Path(os.environ.get("EXAMGEN_EXAM_DATA", str(BASE_DIR / "exam_data.json")))
def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    # Important: backslash first
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("$", r"\$")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s
def sanitize_mcq(q):
    return {
        "stem": latex_escape(q.get("stem", "")),
        "options": [latex_escape(x) for x in q.get("options", ["", "", "", ""])],
        "marks": q.get("marks", 2),
    }

def sanitize_simple(q):
    return {
        "stem": latex_escape(q.get("stem", "")),
        "marks": q.get("marks", 1),
    }
def main():
    # 1. 读取 exam_data.json
    with open(BASE_DIR / "exam_data.json", "r", encoding="utf-8") as f:
        exam_data = json.load(f)

    # 2. 拆数据（给 LaTeX 用）
    meta = exam_data["meta"]
    sections = exam_data["sections"]

    mcq_questions_raw = sections.get("mcq", [])
    saq_questions_raw = sections.get("saq", [])
    lq_questions_raw  = sections.get("lq", [])

    mcq_questions = [sanitize_mcq(q) for q in mcq_questions_raw]
    saq_questions = [sanitize_simple(q) for q in saq_questions_raw]
    lq_questions  = [sanitize_simple(q) for q in lq_questions_raw]

    context = {
        "exam_title": meta["exam_title"],
        "course_code": meta["course_code"],
        "course_name": meta["course_name"],
        "exam_date": meta["exam_date"],
        "mcq_questions": mcq_questions,
        "saq_questions": saq_questions,
        "lq_questions": lq_questions,
        "mcq_count": len(mcq_questions),
        "saq_count": len(saq_questions),
    }

    # 3. 初始化 Jinja2（关键）
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=False,
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((#",
        comment_end_string="#))",
    )


    # 4. 加载模板（⚠️ 你之前缺的就是这一步）
    template = env.get_template("exam_v1.tex")

    # 5. 渲染
    rendered_tex = template.render(context)

    # 6. 输出 .tex 文件
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_tex_path = OUTPUT_DIR / "exam_filled.tex"
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write(rendered_tex)

    print(f"Rendered LaTeX written to {output_tex_path}")

if __name__ == "__main__":
    main()
