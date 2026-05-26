import joblib
import numpy as np

# =========================
# CHARGEMENT DU MEILLEUR MODELE
# =========================
model = joblib.load("best_occupancy_model.pkl")

print("✔ Modèle chargé avec succès")

# =========================
# ENTREES CAPTEURS
# =========================
print("\nEntrez les valeurs des capteurs :")

S1_Light = float(input("S1_Light: "))
S2_Light = float(input("S2_Light: "))
S3_Light = float(input("S3_Light: "))
S4_Light = float(input("S4_Light: "))

S5_CO2 = float(input("S5_CO2: "))
S5_CO2_Slope = float(input("S5_CO2_Slope: "))

S6_PIR = float(input("S6_PIR: "))
S7_PIR = float(input("S7_PIR: "))

Temperature = float(input("Temperature: "))
Sound = float(input("Sound: "))

# =========================
# CREATION DU VECTOR D'ENTREE
# ⚠️ ORDRE EXACT DU TRAINING
# =========================
X_new = np.array([[
    S1_Light,
    S2_Light,
    S3_Light,
    S4_Light,
    S5_CO2,
    S5_CO2_Slope,
    S6_PIR,
    S7_PIR,
    Temperature,
    Sound
]])

# =========================
# PREDICTION
# =========================
prediction = model.predict(X_new)[0]

print("\n====================")
print("RESULTAT FINAL")
print("====================")
print("Nombre d'occupants :", prediction)
