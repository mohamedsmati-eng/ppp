"""
ÉTAPE 4 : DATA CLEANING COMPLET
Vérification, nettoyage et structure des données
"""

import pandas as pd
import numpy as np

print("="*80)
print("ÉTAPE 4 : DATA CLEANING COMPLET")
print("="*80)

# Charger les données
df = pd.read_csv('donnees_moyennes_calculees.csv')
print(f"\nDonnées initiales : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ============================================================================
# 1. VALEURS MANQUANTES
# ============================================================================
print("\n" + "="*80)
print("1. VÉRIFICATION DES VALEURS MANQUANTES")
print("="*80)

valeurs_manquantes = df.isnull().sum()
print("\nValeurs manquantes par colonne :")
print(valeurs_manquantes)

if valeurs_manquantes.sum() > 0:
    print(f"\n✓ Total valeurs manquantes : {valeurs_manquantes.sum()}")
    print("Suppression des lignes avec valeurs manquantes...")
    df = df.dropna()
    print(f"✓ Nouvelles dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
else:
    print("✓ Pas de valeurs manquantes !")

# ============================================================================
# 2. DOUBLONS EXACTS
# ============================================================================
print("\n" + "="*80)
print("2. VÉRIFICATION DES DOUBLONS EXACTS")
print("="*80)

doublons = df.duplicated().sum()
print(f"\nNombre de lignes dupliquées (exactement identiques) : {doublons}")

if doublons > 0:
    print(f"Suppression des {doublons} doublons...")
    df = df.drop_duplicates()
    print(f"✓ Nouvelles dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
else:
    print("✓ Pas de doublons exacts !")

# ============================================================================
# 3. TYPES DE DONNÉES
# ============================================================================
print("\n" + "="*80)
print("3. VÉRIFICATION DES TYPES DE DONNÉES")
print("="*80)

print("\nTypes actuels :")
print(df.dtypes)

# S'assurer que tous les capteurs sont numériques
for col in df.columns:
    if col != 'Room_Occupancy_Count':
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"⚠ {col} n'est pas numérique, conversion...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
# La cible doit être entière (0, 1, 2, 3)
df['Room_Occupancy_Count'] = df['Room_Occupancy_Count'].astype(int)

print("\nTypes après conversion :")
print(df.dtypes)

# ============================================================================
# 4. VALEURS ABERRANTES (OUTLIERS)
# ============================================================================
print("\n" + "="*80)
print("4. DÉTECTION DES VALEURS ABERRANTES")
print("="*80)

colonnes_capteurs = [col for col in df.columns if col != 'Room_Occupancy_Count']

print("\nPlages de valeurs (avant nettoyage) :")
for col in colonnes_capteurs:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col}:")
    print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}, Mean: {df[col].mean():.2f}")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Limites : [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Valeurs aberrantes détectées : {len(outliers)}")

# Décision : Garder les outliers (importants pour la détection d'occupants)
print("\n✓ Les valeurs aberrantes sont conservées (importantes pour le contexte)")

# ============================================================================
# 5. DISTRIBUTION DE LA CIBLE
# ============================================================================
print("\n" + "="*80)
print("5. DISTRIBUTION DE LA VARIABLE CIBLE")
print("="*80)

print("\nDistribution de Room_Occupancy_Count :")
distribution = df['Room_Occupancy_Count'].value_counts().sort_index()
for val in sorted(df['Room_Occupancy_Count'].unique()):
    count = len(df[df['Room_Occupancy_Count'] == val])
    pct = (count / len(df)) * 100
    print(f"  {val} occupant(s) : {count:6d} lignes ({pct:5.1f}%)")

# Vérifier que toutes les classes 0, 1, 2, 3 sont présentes
classes_present = set(df['Room_Occupancy_Count'].unique())
classes_required = {0, 1, 2, 3}
missing_classes = classes_required - classes_present

if missing_classes:
    print(f"\n⚠ Classes manquantes : {missing_classes}")
else:
    print("\n✓ Toutes les classes (0, 1, 2, 3) sont présentes")

# ============================================================================
# 6. STATISTIQUES DESCRIPTIVES
# ============================================================================
print("\n" + "="*80)
print("6. STATISTIQUES DESCRIPTIVES")
print("="*80)

print("\nStatistiques par capteur :")
print(df[colonnes_capteurs].describe())

# ============================================================================
# 7. CORRÉLATIONS
# ============================================================================
print("\n" + "="*80)
print("7. CORRÉLATIONS AVEC LA CIBLE")
print("="*80)

correlations = df.corr()['Room_Occupancy_Count'].drop('Room_Occupancy_Count').sort_values(ascending=False)
print("\nCorrélation de chaque capteur avec Room_Occupancy_Count :")
for col, corr_val in correlations.items():
    print(f"  {col:25} : {corr_val:7.4f}")

# ============================================================================
# 8. STRUCTURE FINALE
# ============================================================================
print("\n" + "="*80)
print("8. STRUCTURE FINALE")
print("="*80)

print(f"\nDimensions finales : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"Colonnes : {df.columns.tolist()}")

print("\n5 premières lignes :")
print(df.head())

print("\n5 dernières lignes :")
print(df.tail())

# ============================================================================
# 9. SAUVEGARDE
# ============================================================================
print("\n" + "="*80)
print("9. SAUVEGARDE")
print("="*80)

df.to_csv('donnees_nettoyees.csv', index=False)
print("\n✓ Données nettoyées sauvegardées : donnees_nettoyees.csv")

# Créer un fichier de résumé
with open('data_cleaning_rapport.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("RAPPORT DE DATA CLEANING\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dimensions finales : {df.shape[0]} lignes × {df.shape[1]} colonnes\n\n")
    f.write("Colonnes :\n")
    for col in df.columns:
        f.write(f"  - {col}\n")
    f.write("\nDistribution de la cible :\n")
    for val in sorted(df['Room_Occupancy_Count'].unique()):
        count = len(df[df['Room_Occupancy_Count'] == val])
        pct = (count / len(df)) * 100
        f.write(f"  {val} occupant(s) : {count} ({pct:.1f}%)\n")
    f.write("\nStatistiques descriptives :\n")
    f.write(df[colonnes_capteurs].describe().to_string())
    f.write("\n\nCorrélations avec la cible :\n")
    for col, corr_val in correlations.items():
        f.write(f"  {col} : {corr_val:.4f}\n")

print("✓ Rapport sauvegardé : data_cleaning_rapport.txt")

print("\n" + "="*80)
print("✓ NETTOYAGE TERMINÉ")
print("="*80 + "\n")