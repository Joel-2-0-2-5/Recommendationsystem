import pandas as pd
import numpy as np
import ast
import joblib
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

# ── Allow frontend to talk to backend ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load saved models and data ────────────────────────────────────
scaler   = joblib.load("model/scaler.pkl")
rf_model = joblib.load("model/rf_model.pkl")
df, df_scaled = joblib.load("model/dataframes.pkl")

df["DisplayIntakes"] = df["DisplayIntakes"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# ── Define what inputs the frontend sends ─────────────────────────
class StudentProfile(BaseModel):
    ielts:               float
    budget:              float
    cbse_score:          float
    backlogs:            int
    intake:              str
    stream:              str = "Any"
    stem_only:           bool = False
    scholarship_needed:  bool = False
    internship_needed:   bool = False
    no_maths:            bool = False
    no_english_test:     bool = False
    fee_waiver:          bool = False

# ── Recommendation logic ──────────────────────────────────────────
rf_features = [
    "QSRanking", "USNewsRanking", "TuitionFeeAUD",
    "EntryRequirementTwelfthOutOf100", "CBSE_RequiredScore",
    "IeltsOverall", "PteScore",
    "IsStemCourse", "ScholarshipAvailable", "InternshipAvailable",
    "AppFeeWaiverAvailable", "WithoutMaths", "WithoutEnglishProficiency",
    "MaxBacklogsAllowed"
]

def recommend(student, raw_df, scaled_df, top_n=10):
    mask = pd.Series([True] * len(raw_df), index=raw_df.index)

    mask &= raw_df["IeltsOverall"]                    <= student.ielts
    mask &= raw_df["TuitionFeeAUD"]                   <= student.budget
    mask &= raw_df["EntryRequirementTwelfthOutOf100"]  <= student.cbse_score
    mask &= raw_df["MaxBacklogsAllowed"]               >= student.backlogs

    if student.intake and student.intake != "Any":
        mask &= raw_df["DisplayIntakes"].apply(lambda x: student.intake in x)
    if student.stem_only:           mask &= raw_df["IsStemCourse"] == 1
    if student.scholarship_needed:  mask &= raw_df["ScholarshipAvailable"] == 1
    if student.internship_needed:   mask &= raw_df["InternshipAvailable"] == 1
    if student.no_maths:            mask &= raw_df["WithoutMaths"] == 1
    if student.no_english_test:     mask &= raw_df["WithoutEnglishProficiency"] == 1
    if student.fee_waiver:          mask &= raw_df["AppFeeWaiverAvailable"] == 1

    filtered_raw    = raw_df[mask].copy()
    filtered_scaled = scaled_df[mask].copy()

    if filtered_raw.empty:
        return pd.DataFrame()

    filtered_raw["score"] = rf_model.predict(filtered_scaled[rf_features])
    return filtered_raw.sort_values("score", ascending=False).head(top_n)

# ── AI Explanation ────────────────────────────────────────────────
def explain_recommendations(student, top_courses):
    client = Groq()

    courses_text = ""
    for i, (_, row) in enumerate(top_courses.head(3).iterrows(), 1):
        courses_text += (
            f"{i}. {row['CourseName']} at {row['universityName']}\n"
            f"   Fee: AUD {row['TuitionFeeAUD']:,.0f} | "
            f"QS Rank: {int(row['QSRanking'])} | "
            f"IELTS Required: {row['IeltsOverall']} | "
            f"Scholarship: {'Yes' if row['ScholarshipAvailable'] else 'No'} | "
            f"Internship: {'Yes' if row['InternshipAvailable'] else 'No'}\n\n"
        )

    prompt = f"""A student has this profile:
- IELTS Score: {student.ielts}
- Budget: AUD {student.budget:,.0f}
- CBSE Score: {student.cbse_score}%
- Backlogs: {student.backlogs}
- Preferred Intake: {student.intake}
- STEM only: {'Yes' if student.stem_only else 'No'}
- Needs Scholarship: {'Yes' if student.scholarship_needed else 'No'}
- Needs Internship: {'Yes' if student.internship_needed else 'No'}

The top recommended courses are:
{courses_text}
In 4-5 sentences, explain why these courses are the best match for this student.
Be specific — mention which profile factors match which course features."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

# ── API Endpoint ──────────────────────────────────────────────────
@app.post("/predict")
def predict(student: StudentProfile):
    results = recommend(student, df, df_scaled, top_n=100)

    if results.empty:
        return {"results": [], "explanation": "No courses matched your profile. Try relaxing some filters."}

    # ── Stream filter applied after recommendation ────────────────
    stream_keywords = {
        "Engineering":                   ["engineering"],
        "Commerce / Business":           ["commerce", "business", "management", "accounting", "marketing", "economics", "entrepreneurship", "finance"],
        "Science":                       ["science", "biotechnology", "actuarial"],
        "Computing / IT / Data Science": ["computer", "computing", "information", "data", "cyber", "software", "it"],
        "Arts / Humanities":             ["arts", "humanities", "social", "media"],
        "Law":                           ["law", "legal", "criminology"],
        "Health / Medicine":             ["health", "medicine", "medical", "nursing", "physiotherapy", "exercise"],
        "Design / Fine Arts":            ["design", "fine arts", "animation", "architecture"],
        "Education":                     ["education", "teaching"],
        "Aviation":                      ["aviation"],
        "Architecture / Planning":       ["architecture", "planning", "construction"],
        "Environmental":                 ["environmental", "sustainability"],
        "Criminology":                   ["criminology", "criminal"],
    }

    if student.stream and student.stream != "Any":
        keywords = stream_keywords.get(student.stream, [])
        if keywords:
            mask = results["CourseName"].str.lower().apply(
                lambda name: any(kw in name for kw in keywords)
            )
            results = results[mask]

    if results.empty:
        return {"results": [], "explanation": "No courses matched your profile for that stream. Try a different stream."}

    output = results[[
        "CourseName", "universityName", "TuitionFeeAUD",
        "QSRanking", "IeltsOverall", "ScholarshipAvailable",
        "InternshipAvailable", "score"
    ]].copy()

    output["TuitionFeeAUD"] = output["TuitionFeeAUD"].round(0)
    output["score"]         = output["score"].round(4)
    output["QSRanking"]     = output["QSRanking"].astype(int)

    explanation = explain_recommendations(student, results)

    return {
        "results":     output.to_dict(orient="records"),
        "explanation": explanation
    }

# ── Chat Endpoint ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:        str
    results_context: str = ""

@app.post("/chat")
def chat(req: ChatRequest):
    client = Groq()

    system_prompt = """You are a friendly study abroad counsellor helping students find the best Australian undergraduate courses. 
You can answer questions about the recommended courses shown to the student, as well as general questions about studying in Australia (visas, cost of living, university life, work rights, etc.).
Keep answers concise, friendly and helpful. If course results are provided, refer to them specifically when relevant."""

    user_message = req.message
    if req.results_context:
        user_message = f"The student is currently viewing these course recommendations:\n{req.results_context}\n\nStudent question: {req.message}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        max_tokens=500
    )
    return {"reply": response.choices[0].message.content}

# ── Per-Course Explanation Endpoint ──────────────────────────────
class CourseExplainRequest(BaseModel):
    course_name:     str
    university_name: str
    tuition_fee:     float
    qs_ranking:      int
    ielts_required:  float
    scholarship:     int
    internship:      int
    student_ielts:   str = ""
    student_budget:  str = ""
    student_cbse:    str = ""

@app.post("/explain")
def explain_course(req: CourseExplainRequest):
    client = Groq()

    prompt = f"""A student is considering this specific course:
- Course: {req.course_name}
- University: {req.university_name}
- Annual Fee: AUD {req.tuition_fee:,.0f}
- QS Ranking: #{req.qs_ranking}
- IELTS Required: {req.ielts_required}
- Scholarship Available: {'Yes' if req.scholarship else 'No'}
- Internship Available: {'Yes' if req.internship else 'No'}

Student profile:
- IELTS Score: {req.student_ielts}
- Budget: AUD {req.student_budget}
- CBSE Score: {req.student_cbse}%

In 5-6 sentences, explain specifically why this course is a strong match for this student. Mention the university's strengths, how the fee fits the budget, career prospects for this field in Australia, and any notable features like scholarship or internship."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )
    return {"explanation": response.choices[0].message.content}

# ── Health check ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "College Recommender API is running"}