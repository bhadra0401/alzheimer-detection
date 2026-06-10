import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# Load Features
X = np.load("features/X_features.npy")
y = np.load("features/y_labels.npy")

print("Features:", X.shape)
print("Labels:", y.shape)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# Models
svm = SVC(
    kernel="rbf",
    probability=True
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

# Train
print("\nTraining SVM...")
svm.fit(X_train, y_train)

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)

print("\nTraining Gradient Boosting...")
gb.fit(X_train, y_train)

# Individual Results
for name, model in [
    ("SVM", svm),
    ("Random Forest", rf),
    ("Gradient Boosting", gb)
]:
    preds = model.predict(X_test)

    print(f"\n{name} Accuracy:")
    print(accuracy_score(y_test, preds))

# Ensemble
ensemble = VotingClassifier(
    estimators=[
        ("svm", svm),
        ("rf", rf),
        ("gb", gb)
    ],
    voting="soft"
)

print("\nTraining Ensemble...")
ensemble.fit(X_train, y_train)

ensemble_preds = ensemble.predict(X_test)

print("\nEnsemble Accuracy:")
print(accuracy_score(y_test, ensemble_preds))

print("\nClassification Report:")
print(classification_report(y_test, ensemble_preds))

# Save Model
import joblib

joblib.dump(
    ensemble,
    "models/alzheimer_ensemble.pkl"
)

print("\nEnsemble Model Saved!")