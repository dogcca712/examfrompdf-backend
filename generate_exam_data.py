import os
import json
import random
import logging
from typing import Optional, List, Tuple
from pathlib import Path
import pdfplumber
from openai import OpenAI  # 如果你用的是 openai 官方 SDK

client = OpenAI()
logger = logging.getLogger(__name__)

def get_pdf_page_count(path: str) -> int:
    """获取PDF的总页数"""
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def extract_text_from_pdf_with_sampling(path: str, target_pages: int, seed: Optional[int] = None) -> str:
    """
    从PDF提取文本，使用分区域+滑动窗口的随机采样策略（方案E）
    
    参数：
    - path: PDF文件路径
    - target_pages: 目标采样页数（按比例分配后）
    - seed: 随机种子（用于可复现，如果为None则完全随机）
    
    采样策略：
    - 如果PDF总页数 <= target_pages：读取所有页
    - 如果PDF总页数 > target_pages：
      - 分成前/中/后三个区域
      - 每个区域按1:1:1比例分配页数
      - 每个区域内部使用滑动窗口随机选择起始位置，然后连续采样
    """
    if seed is not None:
        random.seed(seed)
    
    texts = []
    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        
        # 如果PDF总页数 <= 目标页数，读取所有页
        if total_pages <= target_pages:
            pages_to_read = list(range(total_pages))
            print(f"PDF has {total_pages} pages (<= target {target_pages}), reading all pages")
        else:
            # 分成前/中/后三个区域，每个区域分配 target_pages // 3 页
            pages_per_region = target_pages // 3
            remainder = target_pages % 3  # 余数分配给前区域
            
            # 定义三个区域的边界
            region_size = total_pages // 3
            front_end = region_size
            middle_start = region_size
            middle_end = region_size * 2
            back_start = region_size * 2
            
            pages_to_read = []
            
            # 前区域：随机起始位置，连续采样
            front_pages = pages_per_region + remainder
            if front_end > 0:
                max_start = max(0, front_end - front_pages)
                start = random.randint(0, max_start) if max_start > 0 else 0
                front_selected = list(range(start, min(start + front_pages, front_end)))
                pages_to_read.extend(front_selected)
            
            # 中区域：随机起始位置，连续采样
            if middle_end > middle_start:
                max_start = max(middle_start, middle_end - pages_per_region)
                start = random.randint(middle_start, max_start) if max_start > middle_start else middle_start
                middle_selected = list(range(start, min(start + pages_per_region, middle_end)))
                pages_to_read.extend(middle_selected)
            
            # 后区域：随机起始位置，连续采样
            if total_pages > back_start:
                max_start = max(back_start, total_pages - pages_per_region)
                start = random.randint(back_start, max_start) if max_start > back_start else back_start
                back_selected = list(range(start, min(start + pages_per_region, total_pages)))
                pages_to_read.extend(back_selected)
            
            pages_to_read = sorted(set(pages_to_read))  # 去重并排序
            print(f"PDF has {total_pages} pages, using random sampling (seed={seed}): sampled {len(pages_to_read)} pages from regions [0-{front_end-1}], [{middle_start}-{middle_end-1}], [{back_start}-{total_pages-1}]")
        
        # 提取指定页的文本
        for page_num in pages_to_read:
            page = pdf.pages[page_num]
            # 尝试提取文本，如果失败则尝试其他方法
            text = page.extract_text() or ""
            # 如果提取的文本为空或很少，尝试使用layout模式
            if len(text.strip()) < 10:
                # 尝试提取表格和文本
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = "\n".join([" ".join([str(cell) if cell else "" for cell in row]) for row in table])
                        text += "\n" + table_text
            texts.append(text)
    
    result = "\n\n".join(texts)
    # 如果提取的文本太少，记录警告
    if len(result.strip()) < 50:
        print(f"Warning: Extracted text is very short ({len(result)} chars). PDF may be image-based or encrypted.")
    print(f"Extracted text length: {len(result)} chars from {len(pages_to_read)} pages")
    return result


def extract_text_from_pdf(path: str, max_pages: Optional[int] = None) -> str:
    """
    从PDF提取文本（向后兼容的旧接口）
    
    参数：
    - path: PDF文件路径
    - max_pages: 最大页数限制（如果指定，使用旧逻辑）
    """
    if max_pages is not None:
        # 旧逻辑：直接读取前max_pages页
        texts = []
        with pdfplumber.open(path) as pdf:
            total_pages = min(len(pdf.pages), max_pages)
            for i in range(total_pages):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                if len(text.strip()) < 10:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_text = "\n".join([" ".join([str(cell) if cell else "" for cell in row]) for row in table])
                            text += "\n" + table_text
            texts.append(text)
    return "\n\n".join(texts)
    # 新逻辑：使用智能采样（但这里不推荐使用，应该用extract_text_from_pdf_with_sampling）
    # 为了向后兼容，保留旧逻辑
    return extract_text_from_pdf_with_sampling(path, target_pages=90, seed=None)


def build_prompt(lecture_text: str, mcq_count: int = 10, short_answer_count: int = 3, long_question_count: int = 1, difficulty: str = "medium", special_requests: Optional[str] = None) -> str:
    """
    构造发给 LLM 的 user prompt，要求它输出 exam_data.json 所需的结构。
    这版加入了"禁止出元信息题"的约束，并支持自定义题目数量和难度。
    
    参数：
    - mcq_count: 选择题数量
    - short_answer_count: 简答题数量
    - long_question_count: 论述题数量
    - difficulty: 难度等级 (easy/medium/hard)
    - special_requests: 用户特殊要求（可选）
    """
    # 难度描述
    difficulty_descriptions = {
        "easy": "Easy: Focus on basic concepts, definitions, and straightforward applications. Questions should test fundamental understanding that most students can answer with basic knowledge.",
        "medium": "Medium: Include a mix of basic and intermediate concepts. Questions should require some analysis, application, or synthesis of concepts. Suitable for average students.",
        "hard": "Hard: Emphasize advanced concepts, complex problem-solving, multi-step reasoning, and deep understanding. Questions should challenge even strong students and require critical thinking."
    }
    difficulty_instruction = difficulty_descriptions.get(difficulty, difficulty_descriptions["medium"])
    
    # 构建用户特殊要求的优先级说明
    special_requests_priority = ""
    if special_requests:
        special_requests_priority = f"""
IMPORTANT: User's special request takes absolute priority over all rules below.
User's special request: {special_requests}
All question types (MCQ, SAQ, LQ) must strictly follow this special request, even if it conflicts with default requirements.
"""
    
    return f"""
You are an assistant that creates practice exam papers for university students.

{special_requests_priority if special_requests else ""}
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

Difficulty Level:
{difficulty_instruction}

Question-specific requirements:
- "mcq" must contain EXACTLY {mcq_count} multiple-choice questions.
- Each MCQ must:
  {"  - **IMPORTANT**: Follow the user's special request above regarding question type. If the special request conflicts with the default requirements below, prioritize the special request." if special_requests else ""}
  - {"Otherwise, " if special_requests else ""}focus on a genuine concept or skill from the lecture (e.g. number systems, data representation, machine instructions, operating systems, file system, etc., depending on the lecture content),
  - have exactly 4 options,
  - have only **one clearly correct** option; the other options must be plausible distractors,
  - match the difficulty level specified above.
- "saq" must contain EXACTLY {short_answer_count} short-answer questions.
- Each SAQ must:
  {"  - **IMPORTANT**: Follow the user's special request above regarding question type. If the special request conflicts with the default requirements below, prioritize the special request." if special_requests else ""}
  - {"Otherwise, " if special_requests else ""}require the student to write a few sentences or show small calculations/derivations. Match the difficulty level.
- "lq" must contain EXACTLY {long_question_count} long question(s).
- Each LQ must:
  {"  - **IMPORTANT**: Follow the user's special request above regarding question type. If the special request conflicts with the default requirements below, prioritize the special request." if special_requests else ""}
  - {"Otherwise, " if special_requests else ""}require extended reasoning, explanation, or a multi-step solution. Match the difficulty level.

JSON formatting rules:
- The JSON must be valid and strictly follow the structure above.
- Do NOT include any additional fields beyond those specified.
- Do NOT include explanations, solutions, or answer keys.
- Do NOT wrap the JSON in markdown code fences. Output ONLY raw JSON.

Lecture material:
-----------------
{lecture_text}
-----------------
"""



def build_answer_prompt(exam_data: dict) -> str:
    """
    构建生成答案的提示词
    """
    # 提取题目信息
    mcq_questions = exam_data["sections"]["mcq"]
    saq_questions = exam_data["sections"]["saq"]
    lq_questions = exam_data["sections"]["lq"]
    
    # 构建题目列表（根据语言选择格式）
    mcq_list = []
    saq_list = []
    lq_list = []
    
    # 检测语言：只检测题目内容，不检测元数据
    # 检查所有题目的stem，如果大部分是中文，则认为是中文试卷
    all_question_text = ""
    for q in mcq_questions:
        all_question_text += q.get('stem', '') + " ".join(q.get('options', []))
    for q in saq_questions:
        all_question_text += q.get('stem', '')
    for q in lq_questions:
        all_question_text += q.get('stem', '')
    
    # 统计中文字符数量
    chinese_chars = sum(1 for char in all_question_text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(all_question_text)
    # 如果中文字符占比超过10%，认为是中文试卷
    has_chinese = total_chars > 0 and (chinese_chars / total_chars) > 0.1
    
    if has_chinese:
        # 中文提示词
        for i, q in enumerate(mcq_questions, 1):
            mcq_list.append(f"Q{i}. {q['stem']}\n选项: {', '.join(q['options'])}")
        for i, q in enumerate(saq_questions, 1):
            saq_list.append(f"Q{i}. {q['stem']}")
        for i, q in enumerate(lq_questions, 1):
            lq_list.append(f"Q{i}. {q['stem']}")
        
        prompt = f"""基于刚才生成的考试题目，现在请生成详细的答案解析（Answer Key）。

要求：

1. 对于每道选择题(MCQ)：提供正确答案选项（A/B/C/D）+ 简短解析（解释为什么该选项正确）
   **重要**：每道题的正确答案可能不同，请仔细分析每道题，不要默认选择A。正确答案应该是A、B、C、D中的任意一个，根据题目内容合理选择。

2. 对于每道简答题(Short Answer)：提供参考答案要点 + 评分标准

3. 对于每道论述题(Long Question)：提供完整的参考答案 + 评分细则 + 常见错误提醒

4. 格式清晰，便于教师批改时对照使用

5. 使用中文

请以JSON格式输出，结构如下：
{{
  "mcq": [
    {{
      "correct_option": "A",
      "explanation": "解析内容..."
    }}
  ],
  "saq": [
    {{
      "answer": "参考答案要点...",
      "grading_criteria": "评分标准..."
    }}
  ],
  "lq": [
    {{
      "answer": "完整答案...",
      "grading_criteria": "评分细则...",
      "common_errors": "常见错误提醒..."
    }}
  ]
}}

题目内容：

选择题：
{chr(10).join(mcq_list)}

简答题：
{chr(10).join(saq_list)}

论述题：
{chr(10).join(lq_list)}

请输出严格的JSON格式，不要包含任何额外的文本或markdown代码块。"""
    else:
        # 英文提示词
        for i, q in enumerate(mcq_questions, 1):
            mcq_list.append(f"Q{i}. {q['stem']}\nOptions: {', '.join(q['options'])}")
        for i, q in enumerate(saq_questions, 1):
            saq_list.append(f"Q{i}. {q['stem']}")
        for i, q in enumerate(lq_questions, 1):
            lq_list.append(f"Q{i}. {q['stem']}")
        
        prompt = f"""Based on the exam questions generated earlier, please generate a detailed answer key.

Requirements:

1. For each Multiple Choice Question (MCQ): Provide the correct option (A/B/C/D) + a brief explanation (explain why this option is correct)
   **IMPORTANT**: The correct answer may be different for each question. Please carefully analyze each question and do NOT default to option A. The correct answer should be A, B, C, or D based on the actual question content.

2. For each Short Answer Question: Provide reference answer points + grading criteria

3. For each Long Question: Provide a complete reference answer + detailed grading criteria + common error reminders

4. Clear format, easy for teachers to use when grading

5. Use English

Please output in strict JSON format with the following structure:
{{
  "mcq": [
    {{
      "correct_option": "A",
      "explanation": "Explanation content..."
    }}
  ],
  "saq": [
    {{
      "answer": "Reference answer points...",
      "grading_criteria": "Grading criteria..."
    }}
  ],
  "lq": [
    {{
      "answer": "Complete answer...",
      "grading_criteria": "Detailed grading criteria...",
      "common_errors": "Common error reminders..."
    }}
  ]
}}

Question content:

Multiple Choice Questions:
{chr(10).join(mcq_list)}

Short Answer Questions:
{chr(10).join(saq_list)}

Long Questions:
{chr(10).join(lq_list)}

Please output strict JSON format only, do not include any additional text or markdown code blocks."""
    
    return prompt


def generate_answer_key(exam_data: dict) -> dict:
    """
    生成答案JSON数据
    
    参数：
    - exam_data: 已生成的exam_data字典
    
    返回：
    - 答案字典，包含mcq、saq、lq的答案
    """
    prompt = build_answer_prompt(exam_data)
    
    system_content = "You are an expert exam answer key generator. You always output strict JSON, no extra text."

    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )
    
    raw = response.choices[0].message.content.strip()
    # 清洗markdown代码块
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    
    answer_data = json.loads(raw)
    
    # 验证答案结构
    if "mcq" not in answer_data or "saq" not in answer_data or "lq" not in answer_data:
        raise ValueError("Answer key must contain 'mcq', 'saq', and 'lq' fields")
    
    # 验证答案数量是否匹配
    if len(answer_data["mcq"]) != len(exam_data["sections"]["mcq"]):
        raise ValueError(f"MCQ answer count ({len(answer_data['mcq'])}) doesn't match question count ({len(exam_data['sections']['mcq'])})")
    if len(answer_data["saq"]) != len(exam_data["sections"]["saq"]):
        raise ValueError(f"SAQ answer count ({len(answer_data['saq'])}) doesn't match question count ({len(exam_data['sections']['saq'])})")
    if len(answer_data["lq"]) != len(exam_data["sections"]["lq"]):
        raise ValueError(f"LQ answer count ({len(answer_data['lq'])}) doesn't match question count ({len(exam_data['sections']['lq'])})")
    
    return answer_data


def generate_exam_json(lecture_text: str, mcq_count: int = 10, short_answer_count: int = 3, long_question_count: int = 1, difficulty: str = "medium", special_requests: Optional[str] = None) -> dict:
    """
    生成exam JSON数据
    
    参数：
    - lecture_text: PDF提取的文本
    - mcq_count: 选择题数量
    - short_answer_count: 简答题数量
    - long_question_count: 论述题数量
    - difficulty: 难度等级
    - special_requests: 用户特殊要求（可选）
    """
    prompt = build_prompt(lecture_text, mcq_count, short_answer_count, long_question_count, difficulty, special_requests)

    # 构建 system prompt，包含用户特殊要求
    system_content = "You are an expert exam generator for university-level courses. You always output strict JSON, no extra text."
    if special_requests:
        system_content += f"\n\n用户特殊要求: {special_requests}"

    # 记录提示词信息（用于调试token使用）
    prompt_length = len(prompt)
    system_length = len(system_content)
    lecture_text_length = len(lecture_text)
    print(f"\n=== AI Request Info ===")
    print(f"System prompt length: {system_length} chars")
    print(f"User prompt length: {prompt_length} chars")
    print(f"Lecture text length: {lecture_text_length} chars")
    print(f"Total input (system + user): {system_length + prompt_length} chars")
    print(f"Estimated input tokens: ~{(system_length + prompt_length) // 4} (rough estimate: 1 token ≈ 4 chars)")
    
    # 保存提示词到文件（如果设置了输出目录）
    output_dir = os.environ.get("EXAMGEN_OUTPUT_DIR")
    if output_dir:
        import time
        debug_dir = Path(output_dir).parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        prompt_file = debug_dir / f"prompt_{timestamp}.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write("=== SYSTEM PROMPT ===\n")
            f.write(system_content)
            f.write("\n\n=== USER PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== PROMPT STATS ===\n")
            f.write(f"System: {system_length} chars\n")
            f.write(f"User: {prompt_length} chars\n")
            f.write(f"Lecture text: {lecture_text_length} chars\n")
            f.write(f"Total: {system_length + prompt_length} chars\n")
        print(f"Prompt saved to: {prompt_file}")

    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",  # 默认模型
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    
    # 记录AI响应信息
    response_length = len(raw)
    print(f"AI response length: {response_length} chars")
    print(f"Estimated output tokens: ~{response_length // 4}")
    if hasattr(response, 'usage'):
        print(f"Actual usage - Input tokens: {response.usage.prompt_tokens}, Output tokens: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
    print(f"=== End AI Request Info ===\n")
    
    # 保存AI响应到文件
    if output_dir:
        response_file = debug_dir / f"response_{timestamp}.txt"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write("=== AI RESPONSE ===\n")
            f.write(raw)
            f.write("\n\n=== RESPONSE STATS ===\n")
            f.write(f"Response length: {response_length} chars\n")
            if hasattr(response, 'usage'):
                f.write(f"Input tokens: {response.usage.prompt_tokens}\n")
                f.write(f"Output tokens: {response.usage.completion_tokens}\n")
                f.write(f"Total tokens: {response.usage.total_tokens}\n")
        print(f"Response saved to: {response_file}")
    # 有些模型会不听话输出 ```json ...```，我们简单清洗一下
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()


    return json.loads(raw), raw

def validate_exam_data(exam_data: dict, *, strict_counts: bool = True, expected_mcq: int = 10, expected_saq: int = 3, expected_lq: int = 1) -> list[str]:
    """
    返回错误列表。空列表 = 通过。
    strict_counts=True 表示强制检查期望数量。
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
        if len(mcq) != expected_mcq:
            errors.append(f"MCQ count must be {expected_mcq} (got {len(mcq)})")
        if len(saq) != expected_saq:
            errors.append(f"SAQ count must be {expected_saq} (got {len(saq)})")
        if len(lq) != expected_lq:
            errors.append(f"LQ count must be {expected_lq} (got {len(lq)})")
    else:
        # 允许你测试阶段先放宽
        if not (0 <= len(mcq) <= 50):
            errors.append(f"MCQ count must be between 0 and 50 (got {len(mcq)})")
        if not (0 <= len(saq) <= 20):
            errors.append(f"SAQ count must be between 0 and 20 (got {len(saq)})")
        if not (0 <= len(lq) <= 10):
            errors.append(f"LQ count must be between 0 and 10 (got {len(lq)})")

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
def repair_exam_json(raw_json_text: str, errors: list[str], expected_mcq: int = None, expected_saq: int = None, expected_lq: int = None) -> dict:
    """
    让模型根据错误列表修复 JSON。
    注意：只允许输出 JSON，不允许额外文本。
    
    参数：
    - raw_json_text: 原始JSON文本
    - errors: 错误列表
    - expected_mcq: 期望的MCQ数量（用于数量修复）
    - expected_saq: 期望的SAQ数量
    - expected_lq: 期望的LQ数量
    """
    # 解析当前JSON以获取实际数量
    try:
        current_data = json.loads(raw_json_text)
        current_mcq = len(current_data.get("sections", {}).get("mcq", []))
        current_saq = len(current_data.get("sections", {}).get("saq", []))
        current_lq = len(current_data.get("sections", {}).get("lq", []))
    except:
        current_mcq = current_saq = current_lq = None
    
    # 构建数量修复说明
    count_instructions = []
    if expected_mcq is not None and current_mcq is not None and current_mcq != expected_mcq:
        if current_mcq > expected_mcq:
            count_instructions.append(f"MCQ count: You have {current_mcq} but need {expected_mcq}. REMOVE {current_mcq - expected_mcq} MCQ(s) from the list.")
        else:
            count_instructions.append(f"MCQ count: You have {current_mcq} but need {expected_mcq}. ADD {expected_mcq - current_mcq} more MCQ(s) to the list.")
    
    if expected_saq is not None and current_saq is not None and current_saq != expected_saq:
        if current_saq > expected_saq:
            count_instructions.append(f"SAQ count: You have {current_saq} but need {expected_saq}. REMOVE {current_saq - expected_saq} SAQ(s) from the list.")
        else:
            count_instructions.append(f"SAQ count: You have {current_saq} but need {expected_saq}. ADD {expected_saq - current_saq} more SAQ(s) to the list.")
    
    if expected_lq is not None and current_lq is not None and current_lq != expected_lq:
        if current_lq > expected_lq:
            count_instructions.append(f"LQ count: You have {current_lq} but need {expected_lq}. REMOVE {current_lq - expected_lq} LQ(s) from the list.")
        else:
            count_instructions.append(f"LQ count: You have {current_lq} but need {expected_lq}. ADD {expected_lq - current_lq} more LQ(s) to the list.")
    
    count_instruction_text = "\n".join(count_instructions) if count_instructions else ""
    
    # 构建修复提示（避免在f-string表达式中使用反斜杠）
    count_section = ""
    if count_instruction_text:
        count_section = "CRITICAL COUNT FIXES NEEDED:\n" + count_instruction_text + "\n"
    
    critical_note = ""
    if count_instruction_text:
        critical_note = "CRITICALLY IMPORTANT: Fix the count errors above by removing or adding questions as specified."
    
    repair_prompt = f"""
You previously generated a JSON for an exam, but it failed validation.

Validation errors:
{chr(10).join("- " + e for e in errors)}

{count_section}Your task:
- Output a corrected JSON object that strictly follows the required structure.
- {critical_note}
- Keep the content based on the lecture.
- Do NOT include any extra keys beyond the allowed ones.
- Do NOT wrap in markdown fences.
- Output ONLY raw JSON.

Here is the previous JSON (may be invalid):
{raw_json_text}
"""
    response = client.chat.completions.create(
        model="gpt-5-mini-2025-08-07",  # 备选模型，用于修复任务
        messages=[
            {"role": "system", "content": "You fix JSON to satisfy a strict schema. You MUST fix count errors by adding or removing items. Output only valid JSON."},
            {"role": "user", "content": repair_prompt},
        ],
        # temperature 参数已移除：gpt-5-mini-2025-08-07 只支持默认值 1
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
    
    # 支持命令行参数：pdf_path(s) 和可选的输出路径
    # 格式1: python generate_exam_data.py pdf1.pdf pdf2.pdf ... output.json
    # 格式2: python generate_exam_data.py pdf1.pdf output.json
    # 格式3: python generate_exam_data.py pdf1.pdf (使用默认输出路径)
    
    if len(sys.argv) < 2:
        print("Usage: python generate_exam_data.py <pdf1.pdf> [pdf2.pdf ...] [output.json]")
        sys.exit(1)
    
    # 解析参数：最后一个参数如果是.json，则是输出路径；否则所有参数都是PDF路径
    args = sys.argv[1:]
    if len(args) > 1 and args[-1].endswith('.json'):
        pdf_paths = args[:-1]
        output_path = Path(args[-1])
    else:
        pdf_paths = args
        output_path = Path(os.environ.get("EXAMGEN_OUTPUT_JSON", "exam_data.json"))
    
    # 从环境变量读取exam配置参数
    mcq_count = int(os.environ.get("EXAMGEN_MCQ_COUNT", "10"))
    short_answer_count = int(os.environ.get("EXAMGEN_SHORT_ANSWER_COUNT", "3"))
    long_question_count = int(os.environ.get("EXAMGEN_LONG_QUESTION_COUNT", "1"))
    difficulty = os.environ.get("EXAMGEN_DIFFICULTY", "medium")
    special_requests = os.environ.get("EXAMGEN_SPECIAL_REQUESTS", None)

    print(f"PDF path(s): {', '.join(pdf_paths)}")
    print(f"Output JSON path: {output_path}")
    print(f"Exam config: MCQ={mcq_count}, SAQ={short_answer_count}, LQ={long_question_count}, Difficulty={difficulty}")
    
    # 第一步：统计所有PDF的总页数
    MAX_TOTAL_PAGES = 180  # 总共最多采样180页
    pdf_info = []
    total_pages_all = 0
    
    print(f"\n=== Step 1: Counting pages in all PDFs ===")
    for pdf_path in pdf_paths:
        page_count = get_pdf_page_count(pdf_path)
        pdf_info.append({"path": pdf_path, "total_pages": page_count})
        total_pages_all += page_count
        print(f"  {pdf_path}: {page_count} pages")
    
    print(f"Total pages across all PDFs: {total_pages_all}")
    
    # 第二步：按比例分配180页到各个PDF
    print(f"\n=== Step 2: Allocating {MAX_TOTAL_PAGES} pages proportionally ===")
    allocated_pages = []
    for info in pdf_info:
        if total_pages_all == 0:
            allocated = 0
        else:
            # 按比例分配，四舍五入
            allocated = round(MAX_TOTAL_PAGES * info["total_pages"] / total_pages_all)
        # 确保不超过PDF的实际页数
        allocated = min(allocated, info["total_pages"])
        allocated_pages.append(allocated)
        print(f"  {info['path']}: {info['total_pages']} pages → allocated {allocated} pages")
    
    # 确保总和不超过MAX_TOTAL_PAGES（可能因为四舍五入而超出）
    total_allocated = sum(allocated_pages)
    if total_allocated > MAX_TOTAL_PAGES:
        # 按比例缩减
        scale = MAX_TOTAL_PAGES / total_allocated
        allocated_pages = [max(1, int(p * scale)) for p in allocated_pages]
        # 重新确保不超过PDF实际页数
        for i, info in enumerate(pdf_info):
            allocated_pages[i] = min(allocated_pages[i], info["total_pages"])
        print(f"  Adjusted allocation to fit {MAX_TOTAL_PAGES} pages limit")
    
    # 第三步：使用随机采样提取文本（使用时间戳作为种子，确保每次生成都不同）
    print(f"\n=== Step 3: Extracting text with random sampling (Scheme E) ===")
    all_texts = []
    # 使用时间戳作为随机种子，确保每次生成都不同
    import time
    random_seed = int(time.time() * 1000) % (2**31)  # 使用毫秒级时间戳
    
    for idx, pdf_path in enumerate(pdf_paths):
        target_pages = allocated_pages[idx]
        print(f"\nProcessing: {pdf_path} (target: {target_pages} pages)")
        # 每个PDF使用不同的种子（基于基础种子+索引），确保不同PDF的随机性也不同
        pdf_seed = random_seed + idx * 1000
        lecture_text = extract_text_from_pdf_with_sampling(pdf_path, target_pages, seed=pdf_seed)
        all_texts.append(lecture_text)
        print(f"Extracted {len(lecture_text)} chars from {pdf_path}")
    
    # 合并所有PDF的文本
    combined_text = "\n\n--- PDF分割线 ---\n\n".join(all_texts)
    print(f"\nTotal extracted text length: {len(combined_text)} chars from {len(pdf_paths)} PDF(s)")
    print(f"Total pages sampled: {sum(allocated_pages)} pages (max: {MAX_TOTAL_PAGES})")
    
    # 生成exam JSON
    exam_data, raw = generate_exam_json(
        combined_text,
        mcq_count=mcq_count,
        short_answer_count=short_answer_count,
        long_question_count=long_question_count,
        difficulty=difficulty,
        special_requests=special_requests
    )
    
    # 验证exam数据
    errors = validate_exam_data(exam_data, expected_mcq=mcq_count, expected_saq=short_answer_count, expected_lq=long_question_count)
    
    # 如果验证失败，尝试修复（最多5次，因为数量问题可能需要多次修复）
    max_attempts = 5
    for attempt in range(max_attempts):
        if not errors:
            break

        print("\nVALIDATION FAILED (attempt %d/%d):" % (attempt + 1, max_attempts))
        for e in errors:
            print("-", e)

        if attempt == max_attempts - 1:
            # 最后一次尝试：如果只是数量问题，手动修复
            print("\nLast attempt: Trying manual count fix...")
            try:
                # 手动调整数量（只处理数量过多的情况）
                if len(exam_data["sections"]["mcq"]) > mcq_count:
                    exam_data["sections"]["mcq"] = exam_data["sections"]["mcq"][:mcq_count]
                    print(f"Manually removed {len(exam_data['sections']['mcq']) - mcq_count} extra MCQ(s)")
                
                if len(exam_data["sections"]["saq"]) > short_answer_count:
                    exam_data["sections"]["saq"] = exam_data["sections"]["saq"][:short_answer_count]
                    print(f"Manually removed {len(exam_data['sections']['saq']) - short_answer_count} extra SAQ(s)")
                
                if len(exam_data["sections"]["lq"]) > long_question_count:
                    exam_data["sections"]["lq"] = exam_data["sections"]["lq"][:long_question_count]
                    print(f"Manually removed {len(exam_data['sections']['lq']) - long_question_count} extra LQ(s)")
                
                # 重新验证
                errors = validate_exam_data(exam_data, expected_mcq=mcq_count, expected_saq=short_answer_count, expected_lq=long_question_count)
                if not errors:
                    print("Manual fix successful!")
                    break
            except Exception as e:
                print(f"Manual fix failed: {e}")
            
            raise SystemExit("Too many validation failures.")

        # 让模型修复（传递期望的数量以便更准确地修复）
        exam_data = repair_exam_json(raw, errors, expected_mcq=mcq_count, expected_saq=short_answer_count, expected_lq=long_question_count)
        raw = json.dumps(exam_data, ensure_ascii=False)

        # 重新验证修复后的数据
        errors = validate_exam_data(exam_data, expected_mcq=mcq_count, expected_saq=short_answer_count, expected_lq=long_question_count)

    # 不再自动生成答案，等用户付款后再生成（节省token）
    print("\n=== Skipping answer generation (will be generated after payment) ===")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(exam_data, f, indent=2, ensure_ascii=False)

    print(f"Saved exam_data.json to {output_path} (validated)")


