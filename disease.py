import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Path to audio data
DATA_PATH = "audio_data"

# Emotions
EMOTIONS = ["happy", "sad", "angry", "neutral"]

# Function to extract features (MFCC)
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Load dataset
features = []
labels = []

for emotion in EMOTIONS:
    folder = os.path.join(DATA_PATH, emotion)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            data = extract_features(file_path)
            features.append(data)
            labels.append(emotion)

X = np.array(features)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")

# Save model
joblib.dump(model, "emotion_model.pkl")
print("Model saved as emotion_model.pkl")

# Function to predict emotion from a new audio file
def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    user_file = input("Enter path of audio file (.wav) to predict emotion: ")
    if os.path.exists(user_file):
        emotion = predict_emotion(user_file)
        print("Predicted Emotion:", emotion)
    else:
        print("File not found. Please check the path.")
