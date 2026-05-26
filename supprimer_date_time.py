"""
ÉTAPE 2 : Supprimer les colonnes Date et Time
"""

import pandas as pd

print("="*60)
print("ÉTAPE 2 : SUPPRIMER DATE ET TIME")
print("="*60)

# Charger les données brutes
df = pd.read_csv('donnees_brutes.csv')
print(f"\nDonnées initiales : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"Colonnes : {df.columns.tolist()}")

# Supprimer Date et Time
df = df.drop(['Date', 'Time'], axis=1)
print(f"\n✓ Colonnes 'Date' et 'Time' supprimées")
print(f"✓ Nouvelles dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"✓ Colonnes restantes : {df.columns.tolist()}")

# Sauvegarder
df.to_csv('donnees_sans_date_time.csv', index=False)
print("\n✓ Données sauvegardées dans : donnees_sans_date_time.csv")
