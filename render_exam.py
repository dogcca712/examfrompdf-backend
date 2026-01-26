import json
import re
import random
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
    """
    处理MCQ题目：随机打乱选项顺序，并根据correct_answer_text确定correct_option
    """
    stem = latex_escape(q.get("stem", ""))
    original_options = q.get("options", ["", "", "", ""])
    correct_answer_text = q.get("correct_answer_text", "")
    marks = q.get("marks", 2)
    
    # 转义选项文本
    escaped_options = [latex_escape(x) for x in original_options]
    
    # 找到正确答案在原始选项中的索引
    correct_index = -1
    for i, opt in enumerate(original_options):
        if opt.strip() == correct_answer_text.strip():
            correct_index = i
            break
    
    # 如果找不到正确答案，使用第一个选项作为默认（但会记录警告）
    if correct_index == -1:
        print(f"WARNING: correct_answer_text '{correct_answer_text}' not found in options. Using first option as default.")
        correct_index = 0
    
    # 创建选项索引列表，用于打乱
    indices = list(range(len(escaped_options)))
    # 随机打乱索引
    random.shuffle(indices)
    
    # 根据打乱后的索引重新排列选项
    shuffled_options = [escaped_options[i] for i in indices]
    
    # 找到正确答案在新顺序中的位置
    new_correct_index = indices.index(correct_index)
    # 转换为字母（A=0, B=1, C=2, D=3）
    correct_option = chr(65 + new_correct_index)  # 65是'A'的ASCII码
    
    return {
        "stem": stem,
        "options": shuffled_options,
        "correct_option": correct_option,  # 保存打乱后的正确答案字母
        "marks": marks,
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

    # 处理MCQ题目（随机打乱选项并确定correct_option）
    mcq_questions = []
    mcq_correct_options = []  # 保存打乱后的正确答案，用于后续答案生成
    for q in mcq_questions_raw:
        sanitized = sanitize_mcq(q)
        mcq_questions.append(sanitized)
        mcq_correct_options.append(sanitized.get("correct_option", "A"))
    
    # 将correct_option保存回exam_data.json（用于答案生成）
    for i, correct_option in enumerate(mcq_correct_options):
        if i < len(exam_data["sections"]["mcq"]):
            exam_data["sections"]["mcq"][i]["correct_option"] = correct_option
    
    # 保存更新后的exam_data.json（包含correct_option）
    with open(EXAM_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(exam_data, f, ensure_ascii=False, indent=2)
    
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
