import os
import json
import pdfplumber
from openai import OpenAI  # 如果你用的是 openai 官方 SDK

client = OpenAI()

def extract_text_from_pdf(path: str, max_pages: int = 10) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        num_pages = min(len(pdf.pages), max_pages)
        for i in range(num_pages):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            texts.append(text)
    return "\n\n".join(texts)


def build_prompt(lecture_text: str) -> str:
    """
    构造发给 LLM 的 user prompt，要求它输出 exam_data.json 所需的结构。
    这版加入了“禁止出元信息题”的约束。
    """
    return f"""
You are an assistant that creates practice exam papers for university students.

Below is lecture material text. Your task is to generate a JSON object for a practice exam with the following structure:

{{
  "meta": {{
    "exam_title": "...",
    "course_code": "...",
    "course_name": "...",
    "exam_date": "..."
  }},
  "sections": {{
    "mcq": [
      {{
        "stem": "...",
        "options": ["...", "...", "...", "..."],
        "marks": 2
      }}
    ],
    "saq": [
      {{
        "stem": "...",
        "marks": 6
      }}
    ],
    "lq": [
      {{
        "stem": "...",
        "marks": 20
      }}
    ]
  }}
}}

Important global rules:
- All questions must test **conceptual understanding and problem-solving** based on the technical content of the lecture (e.g. definitions, properties, relationships, examples, reasoning, conversions, etc.).
- DO NOT ask about course logistics, administration, or meta-information.
  - Forbidden topics include (but are not limited to):
    - course code meaning (e.g. "What is INFO1112?")
    - who created or presented the material (lecturer names, acknowledgments)
    - warm-up guides, Canvas, learning resources, or where to find materials
    - lecture outline items as a list (e.g. "Which of the following is in the outline?")
    - week numbers, slide numbers, or any reference to “this lecture”, “today’s lecture”
    - exam date, time, assessment rules, marking schemes, pass marks
- Never ask about trivia or superficial facts that do not help assess understanding of the core technical ideas.
- Avoid “copying entire sentences” from the slides; instead, transform them into questions that require the student to apply or explain the ideas in their own words.

Question-specific requirements:
- "mcq" should contain BETWEEN 2 AND 10 questions for now (we will later enforce exactly 10).
- Each MCQ must:
  - focus on a genuine concept or skill from the lecture (e.g. number systems, data representation, machine instructions, operating systems, file system, etc., depending on the lecture content),
  - have exactly 4 options,
  - have only **one clearly correct** option; the other options must be plausible distractors.
- "saq" should contain 3 short-answer questions that require the student to write a few sentences or show small calculations/derivations.
- "lq" should contain exactly 1 long question that requires extended reasoning, explanation, or a multi-step solution.

JSON formatting rules:
- The JSON must be valid and strictly follow the structure above.
- Do NOT include any additional fields beyond those specified.
- Do NOT include explanations, solutions, or answer keys.
- Do NOT wrap the JSON in markdown code fences. Output ONLY raw JSON.

Lecture material:
-----------------
{lecture_text[:8000]}
-----------------
"""



def generate_exam_json(lecture_text: str) -> dict:
    prompt = build_prompt(lecture_text)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # 或你想用的其它模型
        messages=[
            {
                "role": "system",
                "content": "You are an expert exam generator for university-level courses. You always output strict JSON, no extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    # 有些模型会不听话输出 ```json ...```，我们简单清洗一下
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()


    return json.loads(raw), raw

def validate_exam_data(exam_data: dict, *, strict_counts: bool = True) -> list[str]:
    """
    返回错误列表。空列表 = 通过。
    strict_counts=True 表示强制 10/3/1。
    """
    errors = []

    # ---- basic structure ----
    if not isinstance(exam_data, dict):
        return ["Top-level JSON must be an object/dict"]

    if "meta" not in exam_data or not isinstance(exam_data["meta"], dict):
        errors.append("Missing or invalid 'meta' object")
    if "sections" not in exam_data or not isinstance(exam_data["sections"], dict):
        errors.append("Missing or invalid 'sections' object")

    if errors:
        return errors  # 结构都不对，后面不用继续验

    meta = exam_data["meta"]
    for k in ["exam_title", "course_code", "course_name", "exam_date"]:
        if k not in meta or not isinstance(meta[k], str) or not meta[k].strip():
            errors.append(f"meta.{k} must be a non-empty string")

    sections = exam_data["sections"]
    for sec in ["mcq", "saq", "lq"]:
        if sec not in sections or not isinstance(sections[sec], list):
            errors.append(f"sections.{sec} must be a list")

    if errors:
        return errors

    mcq = sections["mcq"]
    saq = sections["saq"]
    lq = sections["lq"]

    # ---- count constraints ----
    if strict_counts:
        if len(mcq) != 10:
            errors.append(f"MCQ count must be 10 (got {len(mcq)})")
        if len(saq) != 3:
            errors.append(f"SAQ count must be 3 (got {len(saq)})")
        if len(lq) != 1:
            errors.append(f"LQ count must be 1 (got {len(lq)})")
    else:
        # 允许你测试阶段先放宽
        if not (2 <= len(mcq) <= 10):
            errors.append(f"MCQ count must be between 2 and 10 (got {len(mcq)})")
        if len(saq) != 3:
            errors.append(f"SAQ count must be 3 (got {len(saq)})")
        if len(lq) != 1:
            errors.append(f"LQ count must be 1 (got {len(lq)})")

    # ---- per-question validation ----
    def is_int(x): return isinstance(x, int) and not isinstance(x, bool)

    for i, q in enumerate(mcq, start=1):
        if not isinstance(q, dict):
            errors.append(f"MCQ[{i}] must be an object")
            continue
        if "stem" not in q or not isinstance(q["stem"], str) or not q["stem"].strip():
            errors.append(f"MCQ[{i}].stem must be a non-empty string")
        if "options" not in q or not isinstance(q["options"], list) or len(q["options"]) != 4:
            errors.append(f"MCQ[{i}].options must be a list of 4 strings")
        else:
            for j, opt in enumerate(q["options"], start=1):
                if not isinstance(opt, str) or not opt.strip():
                    errors.append(f"MCQ[{i}].options[{j}] must be a non-empty string")
        if "marks" not in q or not is_int(q["marks"]) or q["marks"] <= 0:
            errors.append(f"MCQ[{i}].marks must be a positive integer")

    for i, q in enumerate(saq, start=1):
        if not isinstance(q, dict):
            errors.append(f"SAQ[{i}] must be an object")
            continue
        if "stem" not in q or not isinstance(q["stem"], str) or not q["stem"].strip():
            errors.append(f"SAQ[{i}].stem must be a non-empty string")
        if "marks" not in q or not is_int(q["marks"]) or q["marks"] <= 0:
            errors.append(f"SAQ[{i}].marks must be a positive integer")

    for i, q in enumerate(lq, start=1):
        if not isinstance(q, dict):
            errors.append(f"LQ[{i}] must be an object")
            continue
        if "stem" not in q or not isinstance(q["stem"], str) or not q["stem"].strip():
            errors.append(f"LQ[{i}].stem must be a non-empty string")
        if "marks" not in q or not is_int(q["marks"]) or q["marks"] <= 0:
            errors.append(f"LQ[{i}].marks must be a positive integer")

    # ---- extra-field guard (optional but recommended) ----
    allowed_meta = {"exam_title", "course_code", "course_name", "exam_date"}
    extra_meta = set(meta.keys()) - allowed_meta
    if extra_meta:
        errors.append(f"meta has unexpected keys: {sorted(extra_meta)}")

    allowed_mcq = {"stem", "options", "marks"}  # V1：不让 explanation/answer_index 混进来
    for i, q in enumerate(mcq, start=1):
        if isinstance(q, dict):
            extra = set(q.keys()) - allowed_mcq
            if extra:
                errors.append(f"MCQ[{i}] has unexpected keys: {sorted(extra)}")

    allowed_saq = {"stem", "marks"}
    for i, q in enumerate(saq, start=1):
        if isinstance(q, dict):
            extra = set(q.keys()) - allowed_saq
            if extra:
                errors.append(f"SAQ[{i}] has unexpected keys: {sorted(extra)}")

    allowed_lq = {"stem", "marks"}
    for i, q in enumerate(lq, start=1):
        if isinstance(q, dict):
            extra = set(q.keys()) - allowed_lq
            if extra:
                errors.append(f"LQ[{i}] has unexpected keys: {sorted(extra)}")

    return errors
def repair_exam_json(raw_json_text: str, errors: list[str]) -> dict:
    """
    让模型根据错误列表修复 JSON。
    注意：只允许输出 JSON，不允许额外文本。
    """
    repair_prompt = f"""
You previously generated a JSON for an exam, but it failed validation.

Validation errors:
{chr(10).join("- " + e for e in errors)}

Your task:
- Output a corrected JSON object that strictly follows the required structure.
- Keep the content based on the lecture.
- Do NOT include any extra keys beyond the allowed ones.
- Do NOT wrap in markdown fences.
- Output ONLY raw JSON.

Here is the previous JSON (may be invalid):
{raw_json_text}
"""
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You fix JSON to satisfy a strict schema. Output only valid JSON."},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=1,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 支持命令行参数：pdf_path 和可选的输出路径
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "lecture.pdf"  # 默认值
    
    # 支持环境变量或命令行参数指定输出路径
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = os.environ.get("EXAMGEN_OUTPUT_JSON", "exam_data.json")
    
    output_path = Path(output_path)
    
    text = extract_text_from_pdf(pdf_path)

    print(f"Extracted text length: {len(text)}")
    print(f"PDF path: {pdf_path}")
    print(f"Output JSON path: {output_path}")

    exam_data, raw = generate_exam_json(text)

    for attempt in range(3):  # 最多 3 次（第一次生成 + 2 次修复）
        errors = validate_exam_data(exam_data, strict_counts=True)  # 先放宽，稳定后再改 True
        if not errors:
            break

        print("\nVALIDATION FAILED (attempt %d):" % (attempt + 1))
        for e in errors:
            print("-", e)

        if attempt == 2:
            raise SystemExit("Too many validation failures.")

        # 让模型修复
        exam_data = repair_exam_json(raw, errors)
        raw = json.dumps(exam_data, ensure_ascii=False)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(exam_data, f, indent=2, ensure_ascii=False)

    print(f"Saved exam_data.json to {output_path} (validated)")


