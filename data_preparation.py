import pandas as pd
import numpy as np
import os

# ── Load dataset ───────────────────────────────────────────────────
df = pd.read_excel("data/output_formatted.xlsx")
print(f"Original shape: {df.shape}")


# ── Drop useless columns ───────────────────────────────────────────
single_value_cols = [
    "StudyLevelId", "Studylvl", "CountryId", "universityCountry",
    "CohortCountryCode", "IsOnlineCourse", "CurrencyCode",
    "TuitionFeeCurrency", "ApplicationFeeCurrency", "CommissionCurrency",
    "CommissionMode", "WorkExp", "IeltsRequired", "SatRequired",
    "ActRequired", "GreRequired", "GmatRequired", "DETRequired",
    "FifteenYearsEducation", "ElpAvailable", "EslAvailable",
    "IELTSWaiverScoreCBSE", "IELTSWaiverScoreState", "DETScore",
    "IsMOIWaiver"
]

fully_missing_cols = [
    "IntakesAndDeadlines", "SatScore", "GreScore", "GmatScore",
    "ActScore", "ApplicationDeadlineDetails", "commissionToolTipMessage"
]

internal_cols = [
    "CommissionAmount", "UniversityOrder", "universityLogoExtension",
    "Highlights", "EligibleCountryIdStudyLevelIds", "IntakesClosed",
    "EntryRequirementUG", "EntryRequirementUgOutOf100",
    "EntryRequirementUgOutOf10", "EntryRequirementUgOutOf7",
    "EntryRequirementUgOutOf5", "EntryRequirementUgOutOf4",
    "EnglishMarks12Score", "ApplicationDeadline",
    "ToeflScore", "ToeflNoSectionLessThan", "ESLELPDetail",
    "BacklogRange", "EligibleStateIds", "EligibleCountryIds",
    "UpcomingIntakeDeadLines", "Intakes", "ApplicationMode",
    "IsExtraChoicesOfPrograms", "EntryRequirementTwelfth",
    "Concentration", "WebomatricsNationalRanking", "WebomatricsWorldRanking"
]

cols_to_drop = single_value_cols + fully_missing_cols + internal_cols
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

print(f"After dropping columns: {df.shape}")


# ── Rename columns ─────────────────────────────────────────────────
df.rename(columns={
    "backlog" : "MaxBacklogsAllowed",
    "Amount"  : "TuitionFeeAUD",
    "Name"    : "CourseName"
}, inplace=True)


# ── Fill missing values ───────────────────────────────────────────
df["QSRanking"] = df["QSRanking"].fillna(df["QSRanking"].median())
df["USNewsRanking"] = df["USNewsRanking"].fillna(df["USNewsRanking"].median())
df["EntryRequirementTwelfthOutOf100"] = df["EntryRequirementTwelfthOutOf100"].fillna(df["EntryRequirementTwelfthOutOf100"].median())
df["EntryRequirementTwelfthOutOf10"] = df["EntryRequirementTwelfthOutOf10"].fillna(df["EntryRequirementTwelfthOutOf10"].median())
df["PteScore"] = df["PteScore"].fillna(df["PteScore"].median())
df["PteNoSectionLessThan"] = df["PteNoSectionLessThan"].fillna(df["PteNoSectionLessThan"].median())
df["IeltsNoBandLessThan"] = df["IeltsNoBandLessThan"].fillna(df["IeltsNoBandLessThan"].median())

df["ScholarshipAvailable"] = df["ScholarshipAvailable"].fillna(0)
df["InternshipAvailable"] = df["InternshipAvailable"].fillna(0)
df["MaxBacklogsAllowed"] = df["MaxBacklogsAllowed"].fillna(0)
df["WithoutMaths"] = df["WithoutMaths"].fillna(0)
df["WithoutEnglishProficiency"] = df["WithoutEnglishProficiency"].fillna(0)
df["AppFeeWaiverAvailable"] = df["AppFeeWaiverAvailable"].fillna(0)
df["ToeflRequired"] = df["ToeflRequired"].fillna(0)
df["PteRequired"] = df["PteRequired"].fillna(0)

df["ApplicationFee"] = df["ApplicationFee"].fillna("Not specified")
df["Remarks"] = df["Remarks"].fillna("Not specified")
df["universityCity"] = df["universityCity"].fillna("Not specified")
df["universityState"] = df["universityState"].fillna("Not specified")


# ── Extract CBSE score from EntryRequirement string ────────────────
def extract_cbse_score(text):
    try:
        parts = str(text).split("|")
        for part in parts:
            if "CBSE" in part:
                return float(part.split(":")[1].strip())
    except:
        pass
    return np.nan

df["CBSE_RequiredScore"] = df["EntryRequirement"].apply(extract_cbse_score)
df["CBSE_RequiredScore"].fillna(df["CBSE_RequiredScore"].median(), inplace=True)
df.drop(columns=["EntryRequirement"], inplace=True)


# ── Parse DisplayIntakes into a list ──────────────────────────────
df["DisplayIntakes"] = df["DisplayIntakes"].apply(
    lambda x: [i.strip() for i in str(x).split(",")] if pd.notna(x) else []
)


# ── Convert boolean columns to 0/1 ────────────────────────────────
bool_cols = [
    "IsStemCourse", "ScholarshipAvailable", "InternshipAvailable",
    "AppFeeWaiverAvailable", "WithoutMaths", "WithoutEnglishProficiency",
    "ToeflRequired", "PteRequired"
]
for col in bool_cols:
    df[col] = df[col].astype(int)


# ── Save cleaned data ──────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_data.csv", index=False)

print(f"Final shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("✅ Saved to data/cleaned_data.csv")

