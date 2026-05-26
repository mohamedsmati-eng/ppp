"""
AI-Based Load Management - Model Training

Objectif :
Prédire le nombre d'occupants (0–3)
et sélectionner le meilleur modèle parmi :
- Logistic Regression
- Random Forest
- Gradient Boosting
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# 1. CHARGEMENT DATA
# =========================
TRAIN_FILE = "train_set.csv"
TEST_FILE = "test_set.csv"
TARGET = "Room_Occupancy_Count"

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

X_train = train.drop(TARGET, axis=1)
y_train = train[TARGET]

X_test = test.drop(TARGET, axis=1)
y_test = test[TARGET]

print("Dataset chargé ✔")
print("Train shape :", X_train.shape)
print("Test shape  :", X_test.shape)
print("Classes :", sorted(y_train.unique()))

# =========================
# 2. MODELES
# =========================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=10,              # anti overfitting
        min_samples_split=5,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        random_state=42
    )
}

# =========================
# 3. ENTRAINEMENT + EVALUATION
# =========================
results = []

best_model = None
best_name = None
best_f1 = 0

print("\n==============================")
print("COMPARAISON DES MODELES")
print("==============================")

for name, model in models.items():

    print(f"\n--- {name} ---")

    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy :", round(acc, 4))
    print("F1-score :", round(f1, 4))

    # Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-score": f1
    })

    # Best model selection
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# =========================
# 4. RESULTATS GLOBAUX
# =========================
results_df = pd.DataFrame(results)
print("\n==============================")
print("COMPARAISON FINALE")
print("==============================")
print(results_df)

# =========================
# 5. FEATURE IMPORTANCE
# =========================
if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    print("\n==============================")
    print("IMPORTANCE DES VARIABLES")
    print("==============================")
    print(importance)

    # plot
    plt.figure(figsize=(8,5))
    plt.barh(importance["Feature"], importance["Importance"])
    plt.title("Feature Importance (Best Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

# =========================
# 6. SAUVEGARDE MODELE
# =========================
MODEL_FILE = "best_occupancy_model.pkl"
joblib.dump(best_model, MODEL_FILE)

print("\n==============================")
print("MODELE SELECTIONNE")
print("==============================")
print("Best Model :", best_name)
print("F1-score   :", round(best_f1, 4))
print("Saved as   :", MODEL_FILE)

print("\n=== INTERPRETABILITE DU MEILLEUR MODELE ===")


print("\n✔ Training finished successfully")