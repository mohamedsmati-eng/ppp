"""
ÉTAPE 1 : Charger et explorer le dataset
"""

import pandas as pd

print("="*60) # pour séparer visuellement les étapes 
print("ÉTAPE 1 : CHARGER LE DATASET")
print("="*60)

# Charger le fichier
df = pd.read_csv('room_occupancy.csv')

print(f"\n✓ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print("\nNoms des colonnes :")
print(df.columns.tolist())

print("\n5 premières lignes :")
print(df.head())

print("\nTypes de données :")
print(df.dtypes)

print("\nStatistiques générales :")
print(df.describe())

# Sauvegarder pour l'étape suivante
df.to_csv('donnees_brutes.csv', index=False)
print("\n✓ Données sauvegardées dans : donnees_brutes.csv")