"""
01_split_train_test.py
Sépare les données nettoyées en train et test, sauvegarde les deux fichiers.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Paramètres
INPUT = 'donnees_nettoyees.csv'
TRAIN_OUT = 'train_set.csv'
TEST_OUT = 'test_set.csv'
TEST_SIZE = 0.2  # 20% pour le test
RANDOM_STATE = 42

# Chargement des données
df = pd.read_csv(INPUT)
print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Séparation features / cible
X = df.drop('Room_Occupancy_Count', axis=1)
y = df['Room_Occupancy_Count']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Fusionner pour sauvegarder
train = X_train.copy()
train['Room_Occupancy_Count'] = y_train
test = X_test.copy()
test['Room_Occupancy_Count'] = y_test

train.to_csv(TRAIN_OUT, index=False)
test.to_csv(TEST_OUT, index=False)

print(f"Train set : {train.shape[0]} lignes → {TRAIN_OUT}")
print(f"Test set  : {test.shape[0]} lignes → {TEST_OUT}")

# Vérification de la distribution des classes
print("\nDistribution des classes (train) :")
print(train['Room_Occupancy_Count'].value_counts(normalize=True).sort_index())
print("\nDistribution des classes (test) :")
print(test['Room_Occupancy_Count'].value_counts(normalize=True).sort_index())