import os
import json
import datetime
from typing import Dict, List
import re

import gradio as gr
from google.cloud import storage, bigquery, aiplatform
from vertexai.generative_models import GenerativeModel, Part

# Configuration
PROJECT_ID = "solverchecker"
LOCATION = "us-central1"
BUCKET_NAME = "solver_checker"
DATASET_ID = "solverchecker"
ANSWER_SHEET_TABLE = "answer_sheet"
ANSWER_KEY_TABLE = "answer_keys"
PROCESSED_RESULTS_TABLE = "processed_results"
MODEL_ID = "gemini-2.5-pro"

aiplatform.init(project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)
gcs_client = storage.Client()

# ========== UTILITY FUNCTIONS ==========

def roman_to_int(s: str) -> str:
    m = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,
         'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
    s = s.upper().strip()
    i, n = 0, 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in m:
            n += m[s[i:i+2]]; i += 2
        else:
            n += m.get(s[i], 0); i += 1
    return str(n)

def letter_to_similar_number(c: str) -> str:
    mp = {'O':'0','o':'0','I':'1','l':'1','Z':'2','z':'2',
          'E':'3','e':'3','A':'4','a':'4','S':'5','s':'5',
          'G':'6','g':'6','T':'7','t':'7','B':'8','b':'8',
          'Q':'9','q':'9'}
    return mp.get(c, c)

def extract_batch_suffix(b: str) -> str:
    b = b.strip()
    if len(b) >= 2 and b[-2:].isdigit(): return b
    suf = "".join(letter_to_similar_number(c) for c in b[-2:])
    if suf.isdigit(): return b[:-2] + suf
    return b + "01"

def extract_grade_from_batch(b: str) -> str:
    m = re.search(r'(\d+)', b)
    return m.group(1) if m else "9"

def first_word_lower(s: str) -> str:
    return s.strip().split()[0].lower()

def escape_json(j: Dict) -> str:
    s = json.dumps(j, ensure_ascii=False)
    return s.replace("\\","\\\\").replace("'", "\\'")\
            .replace("\n","\\n").replace("\r","\\r").replace("\t","\\t")

def normalize_json_keys(data: dict) -> dict:
    """Normalize dictionary keys by removing any quote characters."""
    normalized = {}
    for key, value in data.items():
        clean_key = key.strip('"').strip("'")
        normalized[clean_key] = value
    return normalized

def is_subquestion(key: str) -> bool:
    """Check if a key is a sub-question (e.g., '2a', '2i', '2ii')."""
    match = re.match(r'^(\d+)([a-z]+)$', key)
    return bool(match)

def get_parent_key(subq_key: str) -> str:
    """Extract parent question number from sub-question key."""
    match = re.match(r'^(\d+)([a-z]+)$', subq_key)
    if match:
        return match.group(1)
    return subq_key

def get_subquestions_for_parent(parent_key: str, all_keys: set) -> List[str]:
    """Get all sub-questions for a given parent question."""
    subqs = []
    for key in all_keys:
        if is_subquestion(key) and get_parent_key(key) == parent_key:
            subqs.append(key)
    return sorted(subqs)

# ========== AI PROMPTS ==========

ANSWER_SHEET_PROMPT = """
You are an OCR-level precision document transcriber with ZERO error tolerance. Analyze this ANSWER SHEET PDF and return ONLY valid JSON.

TASK 1 – METADATA (first page):
• form_no: digits only
• reg_no: alphanumeric
• test_date: DD-MM-YY format
• subject: exact text
• batch_name: exact text - last 2 characters MUST be digits (convert letters to similar digits if needed)
• organization: extract first word only, lowercase
• grade: extract first digit from batch_name

All characters should be identified and converted accurately and precisely, there is no room for any error.
Return ALL metadata in lowercase.

TASK 2 – ANSWER EXTRACTION WITH SUB-QUESTION DETECTION:

CRITICAL: DISTINGUISHING SUB-QUESTIONS FROM COMBINED ANSWERS

PATTERN 1: SUB-QUESTIONS IN LEFT MARGIN (Each gets its own key)
If you see in the LEFT MARGIN:
  2i [question text]
  2ii [question text]

Then you MUST create SEPARATE entries:
  "2i": "answer text for sub-question i ONLY",
  "2ii": "answer text for sub-question ii ONLY"

NEVER create: "2": "(i) answer i (ii) answer ii"

PATTERN 2: SUB-PARTS WITHIN ANSWER AREA (Same key, combined answer)
If you see in LEFT MARGIN only "2" but in ANSWER AREA the student writes:
  "(i) answer text
   (ii) more answer text"

Then create: "2": "(i) answer text\\n(ii) more answer text"

CRITICAL RULES FOR QUESTION NUMBERING:
1. Question numbers are ONLY found in the LEFT VERTICAL MARGIN LINE
2. Main questions: simple numbers (1, 2, 3, 4...)
3. Sub-questions: append letter/roman directly to parent number with NO separators
   - Examples: 2a, 2b, 2i, 2ii, 3i, 3ii, 4a, 4b
   - NEVER use: 2(a), 2.a, 2-a, 2 a

CRITICAL RULE FOR SUB-QUESTION DETECTION:
4. CHECK THE LEFT MARGIN FIRST:
   - If margin shows "2i" → Create key "2i" with its answer ONLY
   - If margin shows "2ii" → Create key "2ii" with its answer ONLY
   - If margin shows only "2" → Create key "2" with full answer (even if it has "(i)" "(ii)" inside)

CRITICAL RULES FOR MCQ OPTIONS:
5. MCQ options like (a), (b), (c), (1), (2) are NOT question numbers
6. MCQ options appear INSIDE the answer area (right side of margin)
7. If you see an MCQ option in the answer area, include it IN the answer text
8. MCQ options must NEVER become question keys

ANSWER TEXT RULES:
9. Transcribe ALL handwritten text accurately
10. Ignore text with strike-through lines (crossed out)
11. Preserve mathematical notation exactly
12. Include MCQ options at their exact position in the answer text

VERIFICATION CHECKLIST - ASK YOURSELF FOR EACH ANSWER:
- Did I check the LEFT MARGIN for the question number?
- Does the margin show sub-question markers like "2i", "2ii"?
- If YES → Did I create SEPARATE keys ("2i", "2ii") with individual answers?
- If NO → Did I create ONE key ("2") with the complete answer?
- Did I avoid putting "(i)" or "(ii)" markers in the key name?

Sort all keys in ascending alphanumeric order (1, 1a, 1b, 2, 2i, 2ii, 3, 3a, 4...)

All characters should be identified and converted accurately and precisely, there is no room for any error.

OUTPUT ONLY THIS JSON (all metadata lowercase):
{
  "answers": {
    "1": "answer text here",
    "2i": "answer for sub-question i only",
    "2ii": "answer for sub-question ii only",
    "3": "(a) MCQ option with answer text",
    "4a": "sub-question a answer",
    "4b": "sub-question b answer",
    "5": "answer text"
  },
  "metadata": {
    "form_no": "digits",
    "batch_name": "text01",
    "subject": "subject",
    "test_date": "dd-mm-yy",
    "organization": "org",
    "reg_no": "regno",
    "grade": "9"
  }
}
"""

ANSWER_KEY_PROMPT = """
You are an OCR-level precision document transcriber with ZERO error tolerance. Analyze this ANSWER KEY PDF and return ONLY valid JSON.

TASK 1 – METADATA (above horizontal line):
• organization: first word only, lowercase
• grade: TARGET value, convert roman to number, output number only
• batch_name: PHASE value, ensure last 2 characters are digits, lowercase
• subject: exact text, lowercase
• test_date: DD-MM-YY format, lowercase

All characters should be identified and converted accurately and precisely, there is no room for any error.
Return ALL metadata in lowercase.

TASK 2 – ANSWER KEY EXTRACTION:

CRITICAL RULES FOR QUESTION NUMBERING:
1. Question numbers appear in the LEFT area before the answer text
2. Main questions: simple numbers (1, 2, 3, 4...)
3. Sub-questions: letter/roman appended directly to parent number
   - Examples: 2a, 2b, 3i, 3ii, 4a, 4b
   - NEVER include: parentheses, brackets, periods, or separators

CRITICAL RULES FOR MCQ OPTIONS:
4. MCQ options like (a), (b), (c) appearing AFTER a question number are NOT part of the question key
5. These MCQ options must be moved to the START of the answer text field
6. Example: "3 (a) Some answer [2]" → Key: "3", Answer: "(a) Some answer", Max marks: 2

CRITICAL RULES FOR SUB-QUESTIONS:
7. If a question has sub-parts (2i, 2ii, 2iii), NEVER include the parent question "2" as a separate entry
8. Each sub-question gets ONLY its own answer, NOT combined with parent
9. Extract ONLY the individual sub-question answer for each sub-question entry

QUESTION KEY RULES:
10. Question keys contain ONLY: numbers, letters (a-z), roman numerals (i, ii, iii)
11. Question keys NEVER contain: (), [], periods, or MCQ markers

EXTRACTION FORMAT:
12. Maximum marks are in square brackets [X] on the rightmost side
13. Each question/sub-question has its own separate entry with max_marks

QUESTION KEY IDENTIFICATION LOGIC:
- If you see "3 (a) answer text [2]" → Key: "3", Answer: "(a) answer text", Marks: 2
- If you see "3a answer text [1]" → Key: "3a", Answer: "answer text (ONLY for 3a, not parent)", Marks: 1
- If you see "5i answer text [2]" → Key: "5i", Answer: "answer text (ONLY for 5i, not parent)", Marks: 2
- If you see "2. (b) answer text [1]" → Key: "2", Answer: "(b) answer text", Marks: 1

All characters should be identified and converted accurately and precisely, there is no room for any error.

VERIFY EACH QUESTION:
- Remove all periods, parentheses, brackets from question keys
- If MCQ option appears after question number, move it to answer text
- Ensure sub-questions use direct concatenation (2a, NOT 2.a or 2(a))
- For sub-questions: extract ONLY the individual sub-question answer text, not the parent

Sort all keys in ascending alphanumeric order.

OUTPUT ONLY THIS JSON (all metadata lowercase):
{
  "answers": {
    "1": {
      "answer": "complete answer text",
      "max_marks": 1
    },
    "2": {
      "answer": "(a) MCQ option with answer text",
      "max_marks": 1
    },
    "2i": {
      "answer": "answer for sub-question i ONLY",
      "max_marks": 1
    },
    "2ii": {
      "answer": "answer for sub-question ii ONLY",
      "max_marks": 1
    },
    "3a": {
      "answer": "sub-question a answer ONLY",
      "max_marks": 1
    },
    "3b": {
      "answer": "sub-question b answer ONLY",
      "max_marks": 1
    }
  },
  "metadata": {
    "organization": "org",
    "grade": "9",
    "batch_name": "phase01",
    "subject": "subject",
    "test_date": "dd-mm-yy"
  }
}
"""

EVALUATION_PROMPT_TEMPLATE = """
You are a PRECISION EVALUATOR with ZERO tolerance for error. Your task is to compare a student's answer against the answer key with academic rigor and accuracy.

Question Number: {question_no}
Maximum Marks Available: {max_marks}

ANSWER KEY (REFERENCE SOLUTION):
{key_answer}

STUDENT ANSWER (TO BE EVALUATED):
{student_answer}

EVALUATION PROTOCOL:

1. SEMANTIC UNDERSTANDING (70% weight):
   - Does the student demonstrate correct understanding of the concept?
   - Are the core ideas and reasoning aligned with the answer key?
   - Is the logical flow and approach correct?

2. KEY CONCEPTS & ACCURACY (30% weight):
   - Are critical terms, formulas, or facts present?
   - Is the factual accuracy maintained?
   - Are important keywords or concepts mentioned?

3. MCQ HANDLING:
   - If answer key contains MCQ option like "(a)", "(b)", etc., the student answer MUST contain the EXACT same option
   - If student selects different option than answer key, award ZERO marks
   - If both have same option, award full marks

4. MARKING RULES:
   - Marks MUST be multiples of 0.5 ONLY (0, 0.5, 1.0, 1.5, 2.0, 2.5, etc.)
   - Maximum marks cannot exceed {max_marks}
   - Minimum marks is 0
   - Empty, irrelevant, or completely wrong answers get 0 marks
   - Partially correct answers get proportional marks in 0.5 increments
   - Substantially correct answers with minor errors get 80-90% marks
   - Perfect or near-perfect answers get full marks

5. SPECIAL CASES:
   - If student answer is "[Not answered]": award 0 marks
   - If answer key is "[Question not found in answer key]": award 0 marks
   - Ignore minor spelling or grammar errors unless they change meaning
   - Focus on conceptual correctness and presentation

OUTPUT FORMAT - CRITICAL:
You MUST output ONLY a single line of valid JSON with NO markdown formatting, NO code blocks, NO extra text.
Format: {{"marks_obtained": 0.0, "reasoning": "brief explanation here"}}

EXAMPLE OUTPUTS:
{{"marks_obtained": 2.0, "reasoning": "Complete and accurate answer"}}
{{"marks_obtained": 1.0, "reasoning": "Core concept correct but missing key details"}}
{{"marks_obtained": 0.0, "reasoning": "Incorrect concept"}}
{{"marks_obtained": 0.0, "reasoning": "Selected option (b) but correct answer is (a)"}}
"""

# ========== ANSWER SHEET PROCESSING ==========

def process_answer_sheet(pdf_bytes: bytes) -> Dict:
    resp = GenerativeModel(MODEL_ID).generate_content(
        [Part.from_data(pdf_bytes,"application/pdf"), ANSWER_SHEET_PROMPT],
        generation_config={"temperature":0.0,"top_p":0.95,"max_output_tokens":8192}
    )
    raw = resp.text.strip()
    j = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    m = j["metadata"]

    form = m["form_no"].lower().strip()
    reg  = m["reg_no"].lower().strip()
    date = m["test_date"].lower().strip()
    subj = m["subject"].lower().strip()
    batch_raw = m["batch_name"]
    batch = extract_batch_suffix(batch_raw).lower()
    org = first_word_lower(m["organization"])
    grade = extract_grade_from_batch(batch_raw)

    metadata = {
        "form_no": form,
        "reg_no": reg,
        "organization": org,
        "batch_name": batch,
        "subject": subj,
        "test_date": date,
        "grade": grade,
        "document_name": f"{form}.pdf",
        "uri": ""
    }
    return {"answers": j["answers"], "metadata": metadata}

def upload_and_store_sheet(res: Dict, pdf_bytes: bytes):
    meta = res["metadata"]
    path = f"{meta['organization']}/{meta['reg_no']}/{meta['grade']}/{meta['batch_name']}/{meta['subject']}/{meta['test_date']}/answer_sheet"
    doc = meta["document_name"]
    uri = f"gs://{BUCKET_NAME}/{path}/{doc}"

    gcs_client.bucket(BUCKET_NAME).blob(f"{path}/{doc}")\
        .upload_from_string(pdf_bytes, content_type="application/pdf")

    res["metadata"]["uri"] = uri
    processed_at = datetime.datetime.utcnow().isoformat()
    answers_json = escape_json(res["answers"])

    sql = f"""
MERGE `{PROJECT_ID}.{DATASET_ID}.{ANSWER_SHEET_TABLE}` T
USING (
  SELECT
    '{meta['form_no']}' AS form_no,
    '{meta['reg_no']}' AS reg_no,
    '{meta['organization']}' AS organization,
    '{meta['batch_name']}' AS batch_name,
    '{meta['subject']}' AS subject,
    '{meta['test_date']}' AS test_date,
    '{meta['grade']}' AS grade,
    '{doc}' AS document_name,
    '{uri}' AS uri,
    PARSE_JSON('{answers_json}') AS answers,
    TIMESTAMP('{processed_at}') AS processed_at
) S ON T.uri=S.uri
WHEN MATCHED THEN
  UPDATE SET answers=S.answers, processed_at=S.processed_at
WHEN NOT MATCHED THEN
  INSERT (
    form_no, reg_no, organization, batch_name, subject,
    test_date, grade, document_name, uri, answers, processed_at
  )
  VALUES (
    S.form_no, S.reg_no, S.organization, S.batch_name, S.subject,
    S.test_date, S.grade, S.document_name, S.uri, S.answers, S.processed_at
  );
"""
    bq_client.query(sql).result()

# ========== ANSWER KEY PROCESSING ==========

def process_answer_key(pdf_bytes: bytes) -> Dict:
    resp = GenerativeModel(MODEL_ID).generate_content(
        [Part.from_data(pdf_bytes,"application/pdf"), ANSWER_KEY_PROMPT],
        generation_config={"temperature":0.0,"top_p":0.95,"max_output_tokens":8192}
    )
    raw = resp.text.strip()
    j = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    m = j["metadata"]

    org  = first_word_lower(m["organization"])
    batch_raw = m["batch_name"]
    batch = extract_batch_suffix(batch_raw).lower()
    subj = m["subject"].lower().strip()
    date = m["test_date"].lower().strip()
    grd  = str(m["grade"]).strip()
    grade = grd if grd.isdigit() else roman_to_int(grd)
    doc = f"{batch}_{subj}_{date}.pdf"

    regs = get_regions(org)
    if not regs:
        raise ValueError(f"No 6-digit region folders under '{org}'")

    uris = []
    for r in regs:
        path = f"{org}/{r}/{grade}/{batch}/{subj}/{date}/answer_key/{doc}"
        gcs_client.bucket(BUCKET_NAME).blob(path)\
            .upload_from_string(pdf_bytes, content_type="application/pdf")
        uris.append(f"gs://{BUCKET_NAME}/{path}")

    metadata = {
        "organization": org,
        "batch_name": batch,
        "subject": subj,
        "test_date": date,
        "grade": grade,
        "document_name": doc,
        "uris": uris
    }
    return {"answers": j["answers"], "metadata": metadata}

def get_regions(org: str) -> List[str]:
    it = gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=f"{org}/", delimiter="/")
    regions = []
    for page in it.pages:
        for prefix in page.prefixes:
            r = prefix.rstrip("/").split("/")[1]
            if re.fullmatch(r"\d{6}", r): regions.append(r)
    return sorted(regions)

def upsert_answer_key(res: Dict, uris: List[str]):
    tbl = f"{PROJECT_ID}.{DATASET_ID}.{ANSWER_KEY_TABLE}"
    a_json = json.dumps(res["answers"], ensure_ascii=False)
    a_json = a_json.replace("\\","\\\\").replace("'","\\'")\
                   .replace("\n","\\n").replace("\r","\\r").replace("\t","\\t")
    pt = datetime.datetime.utcnow().isoformat()
    m = res["metadata"]
    for uri in uris:
        region = uri.split("/")[1]
        sql = f"""
MERGE `{tbl}` T
USING (
  SELECT
    '{region}' AS region_no,
    '{m['organization']}' AS organization,
    '{m['batch_name']}' AS batch_name,
    '{m['subject']}' AS subject,
    '{m['test_date']}' AS test_date,
    '{m['grade']}' AS grade,
    '{m['document_name']}' AS document_name,
    '{uri}' AS uri,
    PARSE_JSON('{a_json}') AS answers,
    TIMESTAMP('{pt}') AS processed_at
) S ON T.uri=S.uri
WHEN MATCHED THEN
  UPDATE SET answers=S.answers, processed_at=S.processed_at
WHEN NOT MATCHED THEN
  INSERT (
    region_no, organization, batch_name, subject,
    test_date, grade, document_name, uri, answers, processed_at
  )
  VALUES (
    S.region_no, S.organization, S.batch_name, S.subject,
    S.test_date, S.grade, S.document_name, S.uri, S.answers, S.processed_at
  );
"""
        bq_client.query(sql).result()

# ========== BIGQUERY FETCH FUNCTIONS ==========

def fetch_answer_key_from_bigquery(organization: str, batch_name: str, subject: str, test_date: str, grade: str) -> dict:
    """
    Fetch answer key from BigQuery using EXACT values from sheet_metadata.
    CRITICAL: Do NOT transform or modify these values.
    """
    tbl = f"{PROJECT_ID}.{DATASET_ID}.{ANSWER_KEY_TABLE}"
    
    # Use values exactly as provided - only TRIM and LOWER for case-insensitive matching
    sql = f"""
    SELECT answers, organization, batch_name, subject, test_date, grade
    FROM `{tbl}`
    WHERE LOWER(TRIM(organization)) = LOWER(TRIM('{organization}'))
      AND LOWER(TRIM(batch_name)) = LOWER(TRIM('{batch_name}'))
      AND LOWER(TRIM(subject)) = LOWER(TRIM('{subject}'))
      AND LOWER(TRIM(test_date)) = LOWER(TRIM('{test_date}'))
      AND LOWER(TRIM(grade)) = LOWER(TRIM('{grade}'))
    LIMIT 1
    """
    result = bq_client.query(sql).result()
    for row in result:
        answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
        return {
            "answers": answers,
            "metadata": {
                "organization": row["organization"],
                "batch_name": row["batch_name"],
                "subject": row["subject"],
                "test_date": row["test_date"],
                "grade": row["grade"]
            }
        }
    
    # If not found, raise error
    raise ValueError(f"Answer key not found in BigQuery for {organization}/{batch_name}/{subject}/{test_date}/{grade}")

def fetch_answer_sheet_from_bigquery(form_no: str) -> dict:
    """Fetch answer sheet JSON from BigQuery using form_no."""
    tbl = f"{PROJECT_ID}.{DATASET_ID}.{ANSWER_SHEET_TABLE}"
    sql = f"""
    SELECT answers, form_no, reg_no, organization, batch_name, subject, test_date, grade
    FROM `{tbl}`
    WHERE form_no = '{form_no}'
    LIMIT 1
    """
    result = bq_client.query(sql).result()
    for row in result:
        answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
        return {
            "answers": answers,
            "metadata": {
                "form_no": row["form_no"],
                "reg_no": row["reg_no"],
                "organization": row["organization"],
                "batch_name": row["batch_name"],
                "subject": row["subject"],
                "test_date": row["test_date"],
                "grade": row["grade"]
            }
        }
    raise ValueError(f"Answer sheet not found in BigQuery for form_no: {form_no}")

# ========== EVALUATION & MARKING ==========

def evaluate_single_answer(question_no: str, key_answer: str, student_answer: str, max_marks: float) -> dict:
    """Evaluate a single answer with robust JSON parsing for all formats."""
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question_no=question_no,
        key_answer=key_answer,
        student_answer=student_answer,
        max_marks=max_marks
    )
    
    try:
        resp = GenerativeModel(MODEL_ID).generate_content(
            prompt,
            generation_config={"temperature":0.2, "top_p":0.9, "max_output_tokens":512}
        )
        raw = resp.text.strip()
        
        if not raw:
            return {"marks_obtained": 0.0}
        
        # Try multiple parsing strategies
        parsing_attempts = [
            lambda: json.loads(raw),
            lambda: json.loads(raw.split("```json")[1].split("```")[0].strip()) if "```json" in raw else None,
            lambda: json.loads(raw.split("```")[1].strip()) if "```" in raw and "```json" not in raw else None,
            lambda: json.loads(raw[raw.find("{"):raw.rfind("}")+1]) if "{" in raw and "}" in raw else None,
            lambda: json.loads(raw.replace('\\"', '"').replace('\\\\', '\\')),
        ]
        
        result = None
        for attempt in parsing_attempts:
            try:
                parsed = attempt()
                if parsed:
                    result = normalize_json_keys(parsed)
                    break
            except:
                continue
        
        # If parsing succeeded, extract marks
        if result:
            marks = float(result.get("marks_obtained", 0))
            marks = round(marks * 2) / 2
            marks = min(marks, max_marks)
            marks = max(marks, 0)
            return {"marks_obtained": marks}
        
        # Fallback: regex extraction
        marks_match = re.search(r'marks[_\s]*obtained["\s]*:["\s]*([0-9.]+)', raw, re.IGNORECASE)
        if marks_match:
            marks = float(marks_match.group(1))
            marks = round(marks * 2) / 2
            marks = min(marks, max_marks)
            marks = max(marks, 0)
            return {"marks_obtained": marks}
        
        return {"marks_obtained": 0.0}
                
    except Exception as e:
        return {"marks_obtained": 0.0}

def evaluate_answer_sheet(answer_key: dict, answer_sheet: dict, original_sheet_metadata: Dict) -> dict:
    """
    Evaluate entire answer sheet against answer key with proper sub-question handling.
    CRITICAL: Use original_sheet_metadata for result metadata, NOT the corrupted one from BigQuery.
    """
    key_answers = answer_key["answers"]
    student_answers = answer_sheet["answers"]
    test_result = {}
    total_marks_obtained = 0.0
    total_max_marks = 0.0
    
    key_set = set(key_answers.keys())
    student_set = set(student_answers.keys())
    questions_to_evaluate = key_set & student_set
    all_keys_from_key = key_set
    
    for qn in sorted(all_keys_from_key):
        # Skip parent questions if they have sub-questions in the key
        if not is_subquestion(qn):
            subqs = get_subquestions_for_parent(qn, all_keys_from_key)
            if subqs:
                continue
        
        # For this question, get data
        if qn in key_answers:
            key_data = key_answers[qn]
            if isinstance(key_data, dict):
                key_answer = key_data.get("answer", "")
                max_marks = float(key_data.get("max_marks", 1))
            else:
                key_answer = str(key_data)
                max_marks = 1.0
        else:
            key_answer = "[Question not found in answer key]"
            max_marks = 0.0
        
        student_ans = student_answers.get(qn, "[Not answered]")
        
        # Only evaluate if both key and student answer exist
        if qn in questions_to_evaluate and student_ans != "[Not answered]":
            evaluation = evaluate_single_answer(qn, key_answer, student_ans, max_marks)
            marks_obtained = evaluation["marks_obtained"]
        else:
            marks_obtained = 0.0
        
        test_result[qn] = {
            "key_answer": key_answer,
            "student_answer": student_ans,
            "max_marks": max_marks,
            "marks_obtained": marks_obtained
        }
        
        total_marks_obtained += marks_obtained
        total_max_marks += max_marks
    
    percentage = (total_marks_obtained/total_max_marks*100) if total_max_marks else 0.0
    
    # CRITICAL: Use original_sheet_metadata to preserve correct batch_name
    return {
        "batch_name": original_sheet_metadata["batch_name"],
        "subject": original_sheet_metadata["subject"],
        "test_date": original_sheet_metadata["test_date"],
        "form_no": original_sheet_metadata["form_no"],
        "reg_no": original_sheet_metadata["reg_no"],
        "organization": original_sheet_metadata["organization"],
        "grade": original_sheet_metadata["grade"],
        "total_marks_obtained": round(total_marks_obtained, 1),
        "total_max_marks": round(total_max_marks, 1),
        "percentage": round(percentage, 2),
        "test_result": test_result
    }

def store_evaluation_result(result: dict):
    """Store evaluation result in BigQuery."""
    tbl = f"{PROJECT_ID}.{DATASET_ID}.{PROCESSED_RESULTS_TABLE}"
    test_result_json = escape_json(result["test_result"])
    processed_at = datetime.datetime.utcnow().isoformat()
    
    sql = f"""
MERGE `{tbl}` T
USING (
  SELECT
    '{result['batch_name']}' AS batch_name,
    '{result['subject']}' AS subject,
    '{result['test_date']}' AS test_date,
    '{result['form_no']}' AS form_no,
    '{result['reg_no']}' AS reg_no,
    '{result['organization']}' AS organization,
    '{result['grade']}' AS grade,
    {result['total_marks_obtained']} AS total_marks_obtained,
    {result['total_max_marks']} AS total_max_marks,
    {result['percentage']} AS percentage,
    PARSE_JSON('{test_result_json}') AS test_result,
    TIMESTAMP('{processed_at}') AS processed_at
) S
ON T.form_no = S.form_no 
   AND T.batch_name = S.batch_name 
   AND T.subject = S.subject 
   AND T.test_date = S.test_date
WHEN MATCHED THEN
  UPDATE SET 
    total_marks_obtained = S.total_marks_obtained,
    total_max_marks = S.total_max_marks,
    percentage = S.percentage,
    test_result = S.test_result,
    processed_at = S.processed_at
WHEN NOT MATCHED THEN
  INSERT (
    batch_name, subject, test_date, form_no, reg_no, 
    organization, grade, total_marks_obtained, total_max_marks, 
    percentage, test_result, processed_at
  )
  VALUES (
    S.batch_name, S.subject, S.test_date, S.form_no, S.reg_no,
    S.organization, S.grade, S.total_marks_obtained, S.total_max_marks,
    S.percentage, S.test_result, S.processed_at
  );
"""
    bq_client.query(sql).result()

# ========== GRADIO UI ==========

with gr.Blocks() as demo:
    gr.Markdown("# 🎓 SolverChecker Pro - Answer Processing System")
    
    with gr.Tabs():
        with gr.Tab("📝 Answer Sheet"):
            gr.Markdown("### Upload and Process Answer Sheet")
            gr.Markdown("The system will automatically evaluate the answer sheet after processing.")
            inp = gr.File(label="Upload Answer Sheet PDF", type="filepath")
            out = gr.Textbox(label="Processing & Evaluation Results", lines=40)
            btn = gr.Button("Process & Evaluate Sheet", variant="primary", size="lg")
            
            def run_sheet(f):
                if not f: 
                    return json.dumps({"error": "Please upload a PDF"}, indent=2)
                try:
                    # Step 1: Process answer sheet from PDF
                    pdf = open(f,"rb").read()
                    sheet_result = process_answer_sheet(pdf)
                    upload_and_store_sheet(sheet_result, pdf)
                    
                    # CRITICAL: Keep ORIGINAL sheet metadata - do NOT use corrupted version from BigQuery
                    sheet_meta_original = sheet_result["metadata"]
                    form_no = sheet_meta_original["form_no"]
                    
                    try:
                        # Fetch answer sheet from BigQuery (for student answers only)
                        answer_sheet_from_bq = fetch_answer_sheet_from_bigquery(form_no)
                        
                        # Fetch answer key using ORIGINAL metadata
                        answer_key_from_bq = fetch_answer_key_from_bigquery(
                            sheet_meta_original["organization"],
                            sheet_meta_original["batch_name"],
                            sheet_meta_original["subject"],
                            sheet_meta_original["test_date"],
                            sheet_meta_original["grade"]
                        )
                        
                        # Step 3: Evaluate answer sheet
                        # CRITICAL: Pass original_sheet_metadata to preserve correct batch_name in results
                        eval_result = evaluate_answer_sheet(
                            answer_key_from_bq, 
                            answer_sheet_from_bq,
                            sheet_meta_original  # Pass original metadata
                        )
                        
                        # Step 4: Store evaluation results
                        store_evaluation_result(eval_result)
                        
                        # Return success with results
                        return json.dumps({
                            "processing_status": "Success",
                            "sheet_metadata": sheet_meta_original,
                            "evaluation_results": {
                                "form_no": eval_result["form_no"],
                                "reg_no": eval_result["reg_no"],
                                "batch_name": eval_result["batch_name"],
                                "subject": eval_result["subject"],
                                "total_marks_obtained": eval_result["total_marks_obtained"],
                                "total_max_marks": eval_result["total_max_marks"],
                                "percentage": eval_result["percentage"],
                                "test_result": eval_result["test_result"]
                            }
                        }, indent=2)
                    except ValueError as ve:
                        return json.dumps({
                            "processing_status": "Sheet Processed",
                            "sheet_metadata": sheet_meta_original,
                            "evaluation_error": str(ve),
                            "message": "Answer sheet processed and stored successfully, but evaluation failed. Please ensure answer key exists in BigQuery with matching metadata."
                        }, indent=2)
                except Exception as e:
                    return json.dumps({"error": str(e), "type": type(e).__name__}, indent=2)
            
            btn.click(run_sheet, inp, out)

        with gr.Tab("🔑 Answer Key"):
            gr.Markdown("### Upload and Process Answer Key")
            inp2 = gr.File(label="Upload Answer Key PDF", type="filepath")
            out2 = gr.Textbox(label="Results", lines=30)
            btn2 = gr.Button("Process Key", variant="primary", size="lg")
            
            def run_key(f):
                if not f: 
                    return json.dumps({"error": "Please upload a PDF"}, indent=2)
                try:
                    pdf = open(f,"rb").read()
                    res = process_answer_key(pdf)
                    upsert_answer_key(res, res["metadata"]["uris"])
                    return json.dumps({
                        "status": "Success",
                        "message": "Answer key stored in BigQuery",
                        "metadata": res["metadata"],
                        "answers_count": len(res["answers"]),
                        "stored_for_regions": len(res["metadata"]["uris"])
                    }, indent=2)
                except Exception as e:
                    return json.dumps({"error": str(e), "type": type(e).__name__}, indent=2)
            
            btn2.click(run_key, inp2, out2)

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 8080)))
