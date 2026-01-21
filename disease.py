import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("disease.csv")

# Split input (X) and output (y)
X = data.drop("disease", axis=1)
y = data["disease"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Test model accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")

# -----------------------------
# Predict new patient data
# -----------------------------
# FEATURES (TOTAL = 12)
# fatigue, breast_lump, breast_pain, skin_change, nipple_discharge,
# cough, smoking, chest_pain, short_breath, wheezing,
# heart_pain, sweating

new_data = pd.DataFrame(
    [[
        1,  # fatigue
        0,  # breast_lump
        0,  # breast_pain
        0,  # skin_change
        0,  # nipple_discharge
        1,  # cough
        1,  # smoking
        1,  # chest_pain
        1,  # short_breath
        0,  # wheezing
        0,  # heart_pain
        0   # sweating
    ]],
    columns=X.columns
)

result = model.predict(new_data)
print("Predicted Disease:", result[0])
