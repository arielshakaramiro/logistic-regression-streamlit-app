import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# 1. Membuat dataset sederhana
np.random.seed(42)
X = np.random.rand(100, 1) * 10
Y = (X > 5).astype(int).ravel()

# 2. Membagi dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. Training model
model = LogisticRegression()
model.fit(X_train, Y_train)

# 4. Evaluasi
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(classification_report(Y_test, Y_pred))

# 5. Visualisasi
os.makedirs("outputs", exist_ok=True)
plt.scatter(X_test, Y_test, color="blue", label="Actual")
plt.scatter(X_test, Y_pred, color="red", marker="x", label="Predicted")
plt.xlabel("X")
plt.ylabel("Class")
plt.legend()
plt.title("Hasil Logistic Regression")
plt.savefig("outputs/hasil_logistic_regression.png")
plt.close()

# 6. Simpan model
os.makedirs("models", exist_ok=True)
with open("models/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/logistic_model.pkl")
