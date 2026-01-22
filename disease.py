import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("disease.csv")

# Features and target
X = data.drop("disease", axis=1)
y = data["disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\n--- Enter Patient Details ---")

# User input
age = int(input("Age: "))
gender = int(input("Gender (1 = Male, 0 = Female): "))
glucose = int(input("Glucose Level: "))
bp = int(input("Blood Pressure: "))
cholesterol = int(input("Cholesterol Level: "))
breast_lump = int(input("Breast Lump (1 = Yes, 0 = No): "))
chest_pain = int(input("Chest Pain (1 = Yes, 0 = No): "))
fatigue = int(input("Fatigue (1 = Yes, 0 = No): "))
smoking = int(input("Smoking (1 = Yes, 0 = No): "))

# Create input dataframe
user_data = pd.DataFrame([[age, gender, glucose, bp, cholesterol,
                           breast_lump, chest_pain, fatigue, smoking]],
                          columns=X.columns)

# Prediction
result = model.predict(user_data)
print("\nPredicted Disease:", result[0])
