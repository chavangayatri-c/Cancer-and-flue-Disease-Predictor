import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("disease.csv")

# Features and Target
X = data.drop("disease", axis=1)
y = data["disease"]

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Model Accuracy
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy * 100, "%")

# -------------------------------
# USER INPUT SECTION
# -------------------------------
print("\n--- ENTER PATIENT SYMPTOMS ---")
print("Enter 1 for YES, 0 for NO\n")

user_input = []

for symptom in X.columns:
    while True:
        try:
            value = int(input(f"{symptom}: "))
            if value in [0, 1]:
                user_input.append(value)
                break
            else:
                print("Please enter only 0 or 1")
        except ValueError:
            print("Invalid input. Enter 0 or 1 only.")

# Convert to DataFrame (IMPORTANT)
user_df = pd.DataFrame([user_input], columns=X.columns)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(user_df)

print("\n✅ Predicted Disease:", prediction[0])
