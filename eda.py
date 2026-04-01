import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/cleaned_data.csv")

os.makedirs("data/eda_charts", exist_ok=True)

# ── Tuition Fee Distribution ──────────────────────────────────────
plt.figure(figsize=(8, 4))
sns.histplot(df["TuitionFeeAUD"], bins=30, kde=True)
plt.title("Tuition Fee Distribution (AUD)")
plt.xlabel("Annual Tuition Fee (AUD)")
plt.savefig("data/eda_charts/tuition_distribution.png")
plt.close()

# ── IELTS Score Distribution ──────────────────────────────────────
plt.figure(figsize=(8, 4))
sns.histplot(df["IeltsOverall"], bins=15, kde=True)
plt.title("IELTS Overall Score Requirement Distribution")
plt.xlabel("IELTS Overall Score")
plt.savefig("data/eda_charts/ielts_distribution.png")
plt.close()

# ── QS Ranking Distribution ───────────────────────────────────────
plt.figure(figsize=(8, 4))
sns.histplot(df["QSRanking"], bins=30, kde=True)
plt.title("QS Ranking Distribution")
plt.xlabel("QS Ranking")
plt.savefig("data/eda_charts/qs_ranking_distribution.png")
plt.close()

# ── Scholarship vs Internship Availability ────────────────────────
counts = {
    "Scholarship Available": df["ScholarshipAvailable"].sum(),
    "Internship Available": df["InternshipAvailable"].sum(),
    "App Fee Waiver": df["AppFeeWaiverAvailable"].sum(),
    "Without Maths": df["WithoutMaths"].sum(),
    "Without English Test": df["WithoutEnglishProficiency"].sum(),
}
plt.figure(figsize=(8, 4))
plt.bar(counts.keys(), counts.values())
plt.title("Course Feature Counts")
plt.xticks(rotation=15)
plt.savefig("data/eda_charts/feature_counts.png")
plt.close()

# ── Top 10 Universities by Course Count ──────────────────────────
top_unis = df["universityName"].value_counts().head(10)
plt.figure(figsize=(10, 5))
top_unis.plot(kind="barh")
plt.title("Top 10 Universities by Number of Courses")
plt.xlabel("Number of Courses")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("data/eda_charts/top_universities.png")
plt.close()

# ── Intake Month Breakdown ────────────────────────────────────────
from collections import Counter

all_intakes = []
for entry in df["DisplayIntakes"]:
    try:
        intakes = eval(entry) if isinstance(entry, str) else entry
        all_intakes.extend(intakes)
    except:
        pass

intake_counts = Counter(all_intakes)
plt.figure(figsize=(8, 4))
plt.bar(intake_counts.keys(), intake_counts.values())
plt.title("Intake Month Breakdown")
plt.xlabel("Intake Month")
plt.savefig("data/eda_charts/intake_breakdown.png")
plt.close()

print(f"Total courses: {len(df)}")
print(f"Unique universities: {df['universityName'].nunique()}")
print(f"Tuition range: AUD {df['TuitionFeeAUD'].min():,.0f} – {df['TuitionFeeAUD'].max():,.0f}")
print(f"IELTS range: {df['IeltsOverall'].min()} – {df['IeltsOverall'].max()}")
print("✅ Charts saved to data/eda_charts/")