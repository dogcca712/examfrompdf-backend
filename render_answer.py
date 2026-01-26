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

    # 4. 检测语言：检查题目内容，判断是中文还是英文
    all_text = ""
    for q in exam_data.get("sections", {}).get("mcq", []):
        all_text += q.get("stem", "") + " ".join(q.get("options", []))
    for q in exam_data.get("sections", {}).get("saq", []):
        all_text += q.get("stem", "")
    for q in exam_data.get("sections", {}).get("lq", []):
        all_text += q.get("stem", "")
    
    # 统计中文字符占比
    chinese_chars = sum(1 for char in all_text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(all_text)
    is_chinese = total_chars > 0 and (chinese_chars / total_chars) > 0.1
    
    # 根据语言设置标签文本
    if is_chinese:
        labels = {
            "answer_label": "答案",
            "explanation_label": "解析",
            "grading_criteria_label": "评分标准",
            "grading_details_label": "评分细则",
            "common_errors_label": "常见错误提醒",
            "section_mcq": "Section A: 选择题答案",
            "section_saq": "Section B: 简答题答案",
            "section_lq": "Section C: 论述题答案",
            "instructions_title": "评分说明",
            "instructions_item1": "本文档包含详细答案和评分标准。",
            "instructions_item2": "请根据评分标准进行评分。",
            "instructions_item3": "可根据评分标准给予部分分数。",
        }
    else:
        labels = {
            "answer_label": "Answer",
            "explanation_label": "Explanation",
            "grading_criteria_label": "Grading Criteria",
            "grading_details_label": "Grading Details",
            "common_errors_label": "Common Errors",
            "section_mcq": "Section A: Multiple Choice Answers",
            "section_saq": "Section B: Short Answer Questions",
            "section_lq": "Section C: Long Questions",
            "instructions_title": "Instructions for Graders",
            "instructions_item1": "This document contains detailed answers and grading criteria.",
            "instructions_item2": "Use this as a reference when grading student submissions.",
            "instructions_item3": "Partial credit should be awarded based on the grading criteria provided.",
        }

    context = {
        "exam_title": latex_escape(meta["exam_title"]),
        "course_code": latex_escape(meta["course_code"]),
        "course_name": latex_escape(meta["course_name"]),
        "exam_date": latex_escape(meta["exam_date"]),
        "mcq_answers": mcq_answers,
        "saq_answers": saq_answers,
        "lq_answers": lq_answers,
        **labels,  # 添加语言标签
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

