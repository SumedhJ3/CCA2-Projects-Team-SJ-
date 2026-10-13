
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. READ INPUT FROM EXCEL
# ============================================================

input_file = Path("C:/Users/ecl.lab/Desktop/Sumedh/input_data.xlsx")
output_file = Path("comparison_output.xlsx")

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_file.resolve()}\n"
        "Place input_data.xlsx in the same folder as this script."
    )

df = pd.read_excel(input_file)

print("=" * 60)
print("INPUT DATA")
print("=" * 60)
print(df.to_string(index=False))


# ============================================================
# 2. VALIDATE AND CLEAN DATA
# ============================================================

features = ["Age", "Income", "Loan_Amount"]
required_columns = features + ["Category"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required Excel columns: {missing_columns}"
    )

# Convert model features to numeric values.
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing or invalid required values.
df = df.dropna(subset=required_columns).copy()

# Remove blank category labels.
df["Category"] = df["Category"].astype(str).str.strip()
df = df[df["Category"] != ""]

if len(df) < 4:
    raise ValueError(
        "At least 4 valid records are required to train "
        "and test the model. Add more valid Excel records."
    )

if df["Category"].nunique() < 2:
    raise ValueError(
        "The Category column must contain at least "
        "two different categories."
    )

print("\nValid records available:", len(df))


# ============================================================
# 3. TRADITIONAL PROGRAMMING
# ============================================================

def traditional_classification(loan_amount):
    """Classify a loan using fixed, manually written rules."""

    if loan_amount >= 50000:
        return "High"
    elif loan_amount >= 20000:
        return "Medium"
    else:
        return "Low"


df["Traditional_Result"] = df["Loan_Amount"].apply(
    traditional_classification
)


# ============================================================
# 4. MACHINE LEARNING PROGRAMMING
# ============================================================

X = df[features]
y = df["Category"]

# Stratification requires enough examples in every class
# and enough training and testing slots for those classes.
class_counts = y.value_counts()
number_of_classes = y.nunique()
test_count = max(1, int(len(df) * 0.30))
train_count = len(df) - test_count

can_stratify = (
    class_counts.min() >= 2
    and test_count >= number_of_classes
    and train_count >= number_of_classes
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y if can_stratify else None
)

# Create the Decision Tree model.
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3
)

# Train the model.
model.fit(X_train, y_train)


# ============================================================
# 5. EVALUATE THE MACHINE LEARNING MODEL
# ============================================================

ml_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, ml_predictions)

print("\n" + "=" * 60)
print("MACHINE LEARNING RESULTS")
print("=" * 60)

print(f"ML Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        ml_predictions,
        zero_division=0
    )
)


# ============================================================
# 6. PREDICT ALL VALID EXCEL RECORDS
# ============================================================

df["ML_Result"] = model.predict(df[features])


# ============================================================
# 7. COMPARE TRADITIONAL AND ML RESULTS
# ============================================================

df["Results_Match"] = (
    df["Traditional_Result"] == df["ML_Result"]
)

display_columns = [
    col for col in [
        "Customer",
        "Age",
        "Income",
        "Loan_Amount",
        "Category",
        "Traditional_Result",
        "ML_Result",
        "Results_Match"
    ]
    if col in df.columns
]

print("\n" + "=" * 60)
print("TRADITIONAL VS MACHINE LEARNING")
print("=" * 60)

print(df[display_columns].to_string(index=False))


# ============================================================
# 8. SAVE RESULTS TO EXCEL
# ============================================================

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Comparison", index=False)

    # Save test-set predictions separately for evaluation.
    test_results = X_test.copy()
    test_results["Actual_Category"] = y_test
    test_results["Predicted_Category"] = ml_predictions
    test_results.to_excel(
        writer, sheet_name="Test_Predictions", index=True
    )

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)
print("Results saved to:", output_file.resolve())
