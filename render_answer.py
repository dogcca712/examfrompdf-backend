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

def sanitize_mcq_answer(a):
    return {
        "correct_option": latex_escape(a.get("correct_option", "")),
        "explanation": latex_escape(a.get("explanation", "")),
    }

def sanitize_saq_answer(a):
    return {
        "answer": latex_escape(a.get("answer", "")),
        "grading_criteria": latex_escape(a.get("grading_criteria", "")),
    }

def sanitize_lq_answer(a):
    return {
        "answer": latex_escape(a.get("answer", "")),
        "grading_criteria": latex_escape(a.get("grading_criteria", "")),
        "common_errors": latex_escape(a.get("common_errors", "")),
    }

def main():
    # 1. 读取 exam_data.json（支持环境变量指定路径）
    with open(EXAM_DATA_PATH, "r", encoding="utf-8") as f:
        exam_data = json.load(f)

    # 2. 检查是否有答案数据
    if "answers" not in exam_data:
        raise ValueError("No 'answers' field found in exam_data.json. Please generate answers first.")

    # 3. 拆数据（给 LaTeX 用）
    meta = exam_data["meta"]
    answers = exam_data["answers"]

    mcq_answers_raw = answers.get("mcq", [])
    saq_answers_raw = answers.get("saq", [])
    lq_answers_raw = answers.get("lq", [])

    mcq_answers = [sanitize_mcq_answer(a) for a in mcq_answers_raw]
    saq_answers = [sanitize_saq_answer(a) for a in saq_answers_raw]
    lq_answers = [sanitize_lq_answer(a) for a in lq_answers_raw]

    context = {
        "exam_title": latex_escape(meta["exam_title"]),
        "course_code": latex_escape(meta["course_code"]),
        "course_name": latex_escape(meta["course_name"]),
        "exam_date": latex_escape(meta["exam_date"]),
        "mcq_answers": mcq_answers,
        "saq_answers": saq_answers,
        "lq_answers": lq_answers,
    }

    # 4. 初始化 Jinja2（关键）
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

    # 5. 加载模板
    template = env.get_template("answer_v1.tex")

    # 6. 渲染
    rendered_tex = template.render(context)

    # 7. 输出 .tex 文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_tex_path = OUTPUT_DIR / "answer_filled.tex"
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write(rendered_tex)

    print(f"Rendered answer LaTeX written to {output_tex_path}")

if __name__ == "__main__":
    main()

