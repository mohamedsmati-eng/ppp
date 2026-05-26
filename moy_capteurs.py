"""
ÉTAPE 3 : Moyenne des capteurs Temperature et Sound
Remplacer les 4 Temperature par 1 colonne Temperature
Remplacer les 4 Sound par 1 colonne Sound
Garder le reste inchangé
"""

import pandas as pd

print("="*60)
print("ÉTAPE 3 : MOYENNE DES CAPTEURS")
print("="*60)

# Charger les données
df = pd.read_csv('donnees_sans_date_time.csv')
print(f"\nDonnées initiales : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"Colonnes : {df.columns.tolist()}")

# ============================================================================
# MOYENNE DES TEMPERATURES
# ============================================================================
print("\n[1] Moyenne des capteurs Temperature...")

# Trouver les colonnes Temperature (ex: Temperature1, Temperature2, etc)
cols_temp = [col for col in df.columns if 'Temperature' in col or 'Temp' in col]
print(f"✓ Colonnes trouvées : {cols_temp}")

if len(cols_temp) > 1:
    # Calculer la moyenne
    moyenne_temp = df[cols_temp].mean(axis=1)
    print(f"✓ Moyenne calculée : Min={moyenne_temp.min():.2f}, Max={moyenne_temp.max():.2f}")
    
    # Supprimer les colonnes originales
    df = df.drop(cols_temp, axis=1)
    
    # Ajouter la colonne moyenne
    df['Temperature'] = moyenne_temp
    print(f"✓ {len(cols_temp)} colonnes remplacées par 1 colonne 'Temperature'")
else:
    print(f"⚠ Pas assez de colonnes Temperature à moyenner")

# ============================================================================
# MOYENNE DES SOUND
# ============================================================================
print("\n[2] Moyenne des capteurs Sound...")

# Trouver les colonnes Sound (ex: Sound1, Sound2, etc)
cols_sound = [col for col in df.columns if 'Sound' in col]
print(f"✓ Colonnes trouvées : {cols_sound}")

if len(cols_sound) > 1:
    # Calculer la moyenne
    moyenne_sound = df[cols_sound].mean(axis=1)
    print(f"✓ Moyenne calculée : Min={moyenne_sound.min():.2f}, Max={moyenne_sound.max():.2f}")
    
    # Supprimer les colonnes originales
    df = df.drop(cols_sound, axis=1)
    
    # Ajouter la colonne moyenne
    df['Sound'] = moyenne_sound
    print(f"✓ {len(cols_sound)} colonnes remplacées par 1 colonne 'Sound'")
else:
    print(f"⚠ Pas assez de colonnes Sound à moyenner")

# ============================================================================
# RÉSULTAT FINAL
# ============================================================================
print(f"\n✓ Nouvelles dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"✓ Colonnes finales : {df.columns.tolist()}")

print("\n5 premières lignes :")
print(df.head())

# Sauvegarder
df.to_csv('donnees_moyennes_calculees.csv', index=False)
print("\n✓ Données sauvegardées dans : donnees_moyennes_calculees.csv")

print("\n" + "="*60 + "\n")