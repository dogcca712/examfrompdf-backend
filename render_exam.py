import json
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import os

BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = Path(os.environ.get("EXAMGEN_OUTPUT_DIR", str(BASE_DIR / "build")))
EXAM_DATA_PATH = Path(os.environ.get("EXAMGEN_EXAM_DATA", str(BASE_DIR / "exam_data.json")))
def latex_escape(s: str) -> str:
    """
    转义LaTeX特殊字符，但保留数学模式（$...$ 和 $$...$$）不变
    """
    if s is None:
        return ""
    s = str(s)
    
    # 使用占位符保护数学模式（使用不包含特殊字符的占位符）
    math_placeholders = []
    placeholder_counter = 0
    
    # 匹配 $$...$$ (显示数学)
    def replace_display_math(match):
        nonlocal placeholder_counter
        placeholder = f"@@MATHDISPLAY{placeholder_counter}@@"
        placeholder_counter += 1
        math_placeholders.append(match.group(0))  # 保存原始数学内容
        return placeholder
    
    # 匹配 $...$ (行内数学)
    def replace_inline_math(match):
        nonlocal placeholder_counter
        placeholder = f"@@MATHINLINE{placeholder_counter}@@"
        placeholder_counter += 1
        math_placeholders.append(match.group(0))  # 保存原始数学内容
        return placeholder
    
    # 先处理显示数学 $$...$$（避免与行内数学冲突）
    s = re.sub(r'\$\$.*?\$\$', replace_display_math, s, flags=re.DOTALL)
    # 再处理行内数学 $...$（但不能是 $$）
    s = re.sub(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', replace_inline_math, s)
    
    # 转义非数学模式中的特殊字符
    # Important: backslash first
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    
    # 恢复数学模式（不转义）
    for i, math_content in enumerate(math_placeholders):
        s = s.replace(f"@@MATHDISPLAY{i}@@", math_content)
        s = s.replace(f"@@MATHINLINE{i}@@", math_content)
    
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
    # 1. 读取 exam_data.json（支持环境变量指定路径）
    with open(EXAM_DATA_PATH, "r", encoding="utf-8") as f:
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
