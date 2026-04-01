import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import matplotlib.pyplot as plt
import joblib
import os
import ast
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

df = pd.read_csv("data/cleaned_data.csv")

df["DisplayIntakes"] = df["DisplayIntakes"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# ── Columns to normalize ──────────────────────────────────────────
# These are on wildly different scales — normalization prevents any
# single column dominating the score just because its numbers are bigger.
cols_to_scale = [
    "QSRanking",                       # 20–1200
    "USNewsRanking",                   # similar
    "TuitionFeeAUD",                   # 10,000–140,000
    "EntryRequirementTwelfthOutOf100", # 50–100
    "CBSE_RequiredScore",              # similar
    "IeltsOverall",                    # 5.5–8.0
    "PteScore",                        # 36–90
]

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# ── Stage 1: K-Means Clustering ───────────────────────────────────
# Groups courses into clusters by their features.
# When a student submits a profile, we find their closest cluster
# and recommend from it — this is genuine unsupervised ML.

cluster_features = [
    "QSRanking", "USNewsRanking", "TuitionFeeAUD",
    "IeltsOverall", "EntryRequirementTwelfthOutOf100",
    "IsStemCourse", "ScholarshipAvailable", "InternshipAvailable"
]

X_cluster = df_scaled[cluster_features]

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_cluster)
df_scaled["cluster"] = df["cluster"]

sil_score = silhouette_score(X_cluster, df["cluster"])
print(f"K-Means Silhouette Score: {sil_score:.4f}  (closer to 1.0 = better clusters)")

joblib.dump(kmeans, "model/kmeans.pkl")

# ── Generate scoring labels for Random Forest training ────────────
# Lower ranking = better → invert with (1 - value)
# Lower fee = better → invert

ranking_score = (
    (1 - df_scaled["QSRanking"]) * 0.5 +
    (1 - df_scaled["USNewsRanking"]) * 0.5
)

fee_score = 1 - df_scaled["TuitionFeeAUD"]

bonus_score = (
    df["ScholarshipAvailable"] * 0.30 +
    df["InternshipAvailable"]  * 0.30 +
    df["AppFeeWaiverAvailable"]* 0.20 +
    df["IsStemCourse"]         * 0.20
)
if bonus_score.max() > 0:
    bonus_score = bonus_score / bonus_score.max()

df["score"] = (
    ranking_score * 0.40 +
    fee_score     * 0.35 +
    bonus_score   * 0.25
)

# ── Stage 2: Random Forest Regressor ─────────────────────────────
# Learns to predict course score from features.
# Captures non-linear relationships the formula can't — e.g.
# scholarship + low fee together matter more than either alone.

rf_features = [
    "QSRanking", "USNewsRanking", "TuitionFeeAUD",
    "EntryRequirementTwelfthOutOf100", "CBSE_RequiredScore",
    "IeltsOverall", "PteScore",
    "IsStemCourse", "ScholarshipAvailable", "InternshipAvailable",
    "AppFeeWaiverAvailable", "WithoutMaths", "WithoutEnglishProficiency",
    "MaxBacklogsAllowed"
]

X = df_scaled[rf_features]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,  # hyperparameter: number of trees
    max_depth=10,      # hyperparameter: max tree depth
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"Random Forest RMSE: {rmse:.4f}")
print(f"Random Forest R²:   {r2:.4f}  (1.0 = perfect)")

joblib.dump(rf_model, "model/rf_model.pkl")

# ── Feature Importance Plot ───────────────────────────────────────
importances = pd.Series(rf_model.feature_importances_, index=rf_features)
importances.sort_values().plot(kind="barh", figsize=(8, 6))
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("data/eda_charts/feature_importance.png")
plt.close()
print("Feature importance chart saved.")

# ── Recommendation Function ───────────────────────────────────────
def recommend(student, raw_df, scaled_df, top_n=10):
    mask = pd.Series([True] * len(raw_df), index=raw_df.index)

    # Hard eligibility filters
    if student.get("ielts"):
        mask &= raw_df["IeltsOverall"] <= student["ielts"]
    if student.get("budget"):
        mask &= raw_df["TuitionFeeAUD"] <= student["budget"]
    if student.get("cbse_score"):
        mask &= raw_df["EntryRequirementTwelfthOutOf100"] <= student["cbse_score"]
    if student.get("backlogs") is not None:
        mask &= raw_df["MaxBacklogsAllowed"] >= student["backlogs"]
    if student.get("intake"):
        mask &= raw_df["DisplayIntakes"].apply(lambda x: student["intake"] in x)

    # Optional tick-box filters
    if student.get("stem_only"):        mask &= raw_df["IsStemCourse"] == 1
    if student.get("scholarship_needed"): mask &= raw_df["ScholarshipAvailable"] == 1
    if student.get("internship_needed"):  mask &= raw_df["InternshipAvailable"] == 1
    if student.get("no_maths"):           mask &= raw_df["WithoutMaths"] == 1
    if student.get("no_english_test"):    mask &= raw_df["WithoutEnglishProficiency"] == 1
    if student.get("fee_waiver"):         mask &= raw_df["AppFeeWaiverAvailable"] == 1

    filtered_raw    = raw_df[mask].copy()
    filtered_scaled = scaled_df[mask].copy()

    if filtered_raw.empty:
        return pd.DataFrame()

    # Use Random Forest to score filtered courses
    rf_scores = rf_model.predict(filtered_scaled[rf_features])
    filtered_raw["score"] = rf_scores

    return filtered_raw.sort_values("score", ascending=False).head(top_n)

# ── AI Explanation via Groq (free) ────────────────────────────────
def explain_recommendations(student, top_courses):
    client = Groq()  # reads GROQ_API_KEY from environment

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
- IELTS Score: {student.get('ielts', 'Not specified')}
- Budget: AUD {student.get('budget', 'Not specified'):,}
- CBSE Score: {student.get('cbse_score', 'Not specified')}%
- Backlogs: {student.get('backlogs', 0)}
- Preferred Intake: {student.get('intake', 'Any')}
- STEM only: {'Yes' if student.get('stem_only') else 'No'}
- Needs Scholarship: {'Yes' if student.get('scholarship_needed') else 'No'}
- Needs Internship: {'Yes' if student.get('internship_needed') else 'No'}

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

# ── Test with a sample profile ────────────────────────────────────
sample_student = {
    "ielts"              : 6.5,
    "budget"             : 45000,
    "cbse_score"         : 75,
    "backlogs"           : 0,
    "intake"             : "Feb",
    "stem_only"          : False,
    "scholarship_needed" : False,
    "internship_needed"  : False,
    "no_maths"           : False,
    "no_english_test"    : False,
    "fee_waiver"         : False,
}

results = recommend(sample_student, df, df_scaled, top_n=10)

if results.empty:
    print("No courses matched the student profile.")
else:
    print(f"\nTop {len(results)} recommended courses:\n")
    print(results[["CourseName", "universityName", "TuitionFeeAUD",
                    "QSRanking", "IeltsOverall", "score"]].to_string(index=False))

    print("\n── AI Explanation ───────────────────────────────────────────")
    explanation = explain_recommendations(sample_student, results)
    print(explanation)

    results.to_csv("data/sample_recommendations.csv", index=False)
    print("\n✅ Results saved to data/sample_recommendations.csv")

joblib.dump((df, df_scaled), "model/dataframes.pkl")
print("✅ All models saved to model/")