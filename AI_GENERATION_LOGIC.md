# AI 题目生成逻辑说明

## 概述

系统使用 OpenAI GPT-4.1-mini 模型生成考试题目。生成过程分为以下步骤：

1. **PDF文本提取** → 提取前10页文本（最多8000字符）
2. **Prompt构建** → 根据用户配置构建提示词
3. **AI生成** → 调用GPT模型生成JSON格式的题目
4. **验证和修复** → 验证题目数量和质量，如有错误自动修复（最多3次）

---

## 难度级别说明

### Easy（简单）
```
Easy: Focus on basic concepts, definitions, and straightforward applications. 
Questions should test fundamental understanding that most students can answer 
with basic knowledge.
```

**特点：**
- 基础概念和定义
- 直接应用
- 测试基本理解
- 大多数学生都能回答

### Medium（中等）- 默认
```
Medium: Include a mix of basic and intermediate concepts. Questions should require 
some analysis, application, or synthesis of concepts. Suitable for average students.
```

**特点：**
- 基础+中级概念混合
- 需要分析和应用
- 需要综合理解
- 适合普通学生

### Hard（困难）
```
Hard: Emphasize advanced concepts, complex problem-solving, multi-step reasoning, 
and deep understanding. Questions should challenge even strong students and require 
critical thinking.
```

**特点：**
- 高级概念
- 复杂问题解决
- 多步骤推理
- 深度理解
- 需要批判性思维
- 挑战优秀学生

---

## 题目数量要求

### 选择题（MCQ）
- **默认数量**：10题
- **范围**：0-50题
- **要求**：
  - 必须包含 EXACTLY {mcq_count} 道选择题
  - 每题4个选项
  - 只有一个正确答案
  - 其他选项必须是合理的干扰项
  - 必须匹配指定的难度级别

### 简答题（SAQ）
- **默认数量**：3题
- **范围**：0-20题
- **要求**：
  - 必须包含 EXACTLY {short_answer_count} 道简答题
  - 要求学生写几句话或展示小计算/推导
  - 必须匹配指定的难度级别

### 论述题（LQ）
- **默认数量**：1题
- **范围**：0-10题
- **要求**：
  - 必须包含 EXACTLY {long_question_count} 道论述题
  - 需要扩展推理、解释或多步骤解决方案
  - 必须匹配指定的难度级别

---

## 完整 Prompt 示例

### 示例1：Medium难度，默认数量（10 MCQ, 3 SAQ, 1 LQ）

```
You are an assistant that creates practice exam papers for university students.

Below is lecture material text. Your task is to generate a JSON object for a practice exam with the following structure:

{
  "meta": {
    "exam_title": "...",
    "course_code": "...",
    "course_name": "...",
    "exam_date": "..."
  },
  "sections": {
    "mcq": [
      {
        "stem": "...",
        "options": ["...", "...", "...", "..."],
        "marks": 2
      }
    ],
    "saq": [
      {
        "stem": "...",
        "marks": 6
      }
    ],
    "lq": [
      {
        "stem": "...",
        "marks": 20
      }
    ]
  }
}

Important global rules:
- All questions must test **conceptual understanding and problem-solving** based on the technical content of the lecture (e.g. definitions, properties, relationships, examples, reasoning, conversions, etc.).
- DO NOT ask about course logistics, administration, or meta-information.
  - Forbidden topics include (but are not limited to):
    - course code meaning (e.g. "What is INFO1112?")
    - who created or presented the material (lecturer names, acknowledgments)
    - warm-up guides, Canvas, learning resources, or where to find materials
    - lecture outline items as a list (e.g. "Which of the following is in the outline?")
    - week numbers, slide numbers, or any reference to "this lecture", "today's lecture"
    - exam date, time, assessment rules, marking schemes, pass marks
- Never ask about trivia or superficial facts that do not help assess understanding of the core technical ideas.
- Avoid "copying entire sentences" from the slides; instead, transform them into questions that require the student to apply or explain the ideas in their own words.

Difficulty Level:
Medium: Include a mix of basic and intermediate concepts. Questions should require some analysis, application, or synthesis of concepts. Suitable for average students.

Question-specific requirements:
- "mcq" must contain EXACTLY 10 multiple-choice questions.
- Each MCQ must:
  - focus on a genuine concept or skill from the lecture (e.g. number systems, data representation, machine instructions, operating systems, file system, etc., depending on the lecture content),
  - have exactly 4 options,
  - have only **one clearly correct** option; the other options must be plausible distractors,
  - match the difficulty level specified above.
- "saq" must contain EXACTLY 3 short-answer questions that require the student to write a few sentences or show small calculations/derivations. Match the difficulty level.
- "lq" must contain EXACTLY 1 long question(s) that require extended reasoning, explanation, or a multi-step solution. Match the difficulty level.

JSON formatting rules:
- The JSON must be valid and strictly follow the structure above.
- Do NOT include any additional fields beyond those specified.
- Do NOT include explanations, solutions, or answer keys.
- Do NOT wrap the JSON in markdown code fences. Output ONLY raw JSON.

Lecture material:
-----------------
[前8000字符的PDF文本内容]
-----------------
```

### 示例2：Hard难度，自定义数量（5 MCQ, 2 SAQ, 1 LQ）

```
[前面的内容相同...]

Difficulty Level:
Hard: Emphasize advanced concepts, complex problem-solving, multi-step reasoning, and deep understanding. Questions should challenge even strong students and require critical thinking.

Question-specific requirements:
- "mcq" must contain EXACTLY 5 multiple-choice questions.
- Each MCQ must:
  - focus on a genuine concept or skill from the lecture (e.g. number systems, data representation, machine instructions, operating systems, file system, etc., depending on the lecture content),
  - have exactly 4 options,
  - have only **one clearly correct** option; the other options must be plausible distractors,
  - match the difficulty level specified above.
- "saq" must contain EXACTLY 2 short-answer questions that require the student to write a few sentences or show small calculations/derivations. Match the difficulty level.
- "lq" must contain EXACTLY 1 long question(s) that require extended reasoning, explanation, or a multi-step solution. Match the difficulty level.

[后面的内容相同...]
```

---

## AI 模型配置

### 主要生成模型
- **模型**：`gpt-4.1-mini`
- **Temperature**：0.2（较低，确保输出稳定一致）
- **System Message**：`"You are an expert exam generator for university-level courses. You always output strict JSON, no extra text."`

### 修复模型（如果生成失败）
- **模型**：`gpt-5-mini`
- **Temperature**：1.0（较高，允许更多创造性修复）
- **System Message**：`"You fix JSON to satisfy a strict schema. Output only valid JSON."`

---

## 验证和修复流程

1. **第一次生成**：使用主模型生成JSON
2. **验证**：检查题目数量、结构、必填字段
3. **如果失败**：
   - 使用修复模型根据错误列表修复JSON
   - 最多尝试3次（1次生成 + 2次修复）
   - 如果3次都失败，抛出错误

### 验证规则

**严格模式（strict_counts=True）**：
- MCQ数量必须 **完全等于** 期望值
- SAQ数量必须 **完全等于** 期望值
- LQ数量必须 **完全等于** 期望值

**宽松模式（strict_counts=False）**：
- MCQ：0-50题
- SAQ：0-20题
- LQ：0-10题

---

## 禁止的题目类型

AI被明确禁止生成以下类型的题目：

1. **课程元信息**：
   - 课程代码含义（如"What is INFO1112?"）
   - 讲师姓名、致谢
   - 学习资源位置

2. **课程管理**：
   - Canvas、学习平台
   - 课程大纲列表
   - 周数、幻灯片编号
   - "这节课"、"今天的讲座"等引用

3. **考试管理**：
   - 考试日期、时间
   - 评分规则、及格分数
   - 评估规则

4. **浅层事实**：
   - 不帮助评估核心理解的琐事
   - 直接复制幻灯片句子

---

## 数据流

```
用户配置 (前端)
  ↓
FormData: mcq_count, short_answer_count, long_question_count, difficulty
  ↓
后端 /generate 端点
  ↓
exam_config 字典
  ↓
环境变量: EXAMGEN_MCQ_COUNT, EXAMGEN_SHORT_ANSWER_COUNT, etc.
  ↓
generate_exam_data.py
  ↓
build_prompt() → 构建完整prompt
  ↓
generate_exam_json() → 调用GPT API
  ↓
validate_exam_data() → 验证结果
  ↓
repair_exam_json() → 如有错误，修复（最多2次）
  ↓
exam_data.json → 保存到文件
  ↓
render_exam.py → 渲染为LaTeX
  ↓
pdflatex/xelatex → 生成PDF
```

---

## 关键代码位置

- **Prompt构建**：`backend/generate_exam_data.py` → `build_prompt()` (第32行)
- **AI调用**：`backend/generate_exam_data.py` → `generate_exam_json()` (第126行)
- **验证逻辑**：`backend/generate_exam_data.py` → `validate_exam_data()` (第164行)
- **修复逻辑**：`backend/generate_exam_data.py` → `repair_exam_json()` (第281行)

