import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT = 'donnees_nettoyees.csv'
OUT_DIR = 'graphes'
os.makedirs(OUT_DIR, exist_ok=True)
sns.set(style='whitegrid')
plt.rcParams.update({'font.size': 13})

df = pd.read_csv(INPUT)
features = [c for c in df.columns if c != 'Room_Occupancy_Count']

def save_and_show(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    time.sleep(0.2)

# 1. Histogrammes de chaque feature
n = len(features)
cols_per_row = 3
rows = (n + cols_per_row - 1) // cols_per_row
fig, axes = plt.subplots(rows, cols_per_row, figsize=(6*cols_per_row, 4*rows))
axes = axes.flatten()
for i, c in enumerate(features):
    sns.histplot(df[c], bins=30, kde=True, color='steelblue', ax=axes[i])
    axes[i].set_title(f'Histogramme de {c}')
    axes[i].set_xlabel('Valeur')
    axes[i].set_ylabel('Fréquence')
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
save_and_show(fig, '01_histogrammes.png')

# 2. Boxplots de chaque feature
fig, axes = plt.subplots(rows, cols_per_row, figsize=(6*cols_per_row, 4*rows))
axes = axes.flatten()
for i, c in enumerate(features):
    sns.boxplot(y=df[c], color='orange', ax=axes[i])
    axes[i].set_title(f'Boxplot de {c}')
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
save_and_show(fig, '02_boxplots.png')

# 3. Distribution de la cible
fig, ax = plt.subplots(figsize=(7,5))
target_counts = df['Room_Occupancy_Count'].value_counts().sort_index()
sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, palette='crest', ax=ax)
ax.set_xlabel('Nombre d\'occupants')
ax.set_ylabel('Nombre d\'occurrences')
ax.set_title('Distribution de Room_Occupancy_Count')
for i, v in enumerate(target_counts.values):
    ax.text(i, v + max(target_counts.values)*0.01, str(v), ha='center', va='bottom', fontsize=12)
plt.tight_layout()
save_and_show(fig, '03_distribution_cible.png')

# 4. Heatmap de corrélation (features entre elles)
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(1.5*len(features), 1.2*len(features)))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, square=True, linewidths=.5, cbar_kws={'shrink': .7}, ax=ax)
ax.set_title('Matrice de corrélation (features entre elles)')
plt.tight_layout()
save_and_show(fig, '04_correlation_heatmap.png')

# 5. Corrélation avec la cible (barres horizontales)
target_corr = df.corr()['Room_Occupancy_Count'].drop('Room_Occupancy_Count').sort_values()
fig, ax = plt.subplots(figsize=(8, max(4, 0.4*len(target_corr))))
colors = ['tab:green' if v>0 else 'tab:red' for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors)
ax.set_xlabel('Corrélation avec Room_Occupancy_Count')
ax.set_title('Corrélation des capteurs avec la cible')
for i, (name, val) in enumerate(target_corr.items()):
    ax.text(val + (0.02 if val >= 0 else -0.02), i, f'{val:.2f}', va='center', ha='left' if val>=0 else 'right', fontsize=10)
plt.tight_layout()
save_and_show(fig, '05_correlation_with_target.png')

# 6. Boxplots de chaque feature par classe cible
fig, axes = plt.subplots(n, 1, figsize=(8, 3*n))
if n == 1:
    axes = [axes]
for i, c in enumerate(features):
    sns.boxplot(x='Room_Occupancy_Count', y=c, data=df, palette='Set2', ax=axes[i])
    axes[i].set_title(f'{c} par nombre d\'occupants')
plt.tight_layout()
save_and_show(fig, '06_boxplots_by_target.png')

# 7. Scatterplots (stripplots) pour les 3 features les plus corrélées avec la cible
top3 = list(target_corr.abs().sort_values(ascending=False).head(3).index)
for feat in top3:
    fig, ax = plt.subplots(figsize=(8,5))
    sns.stripplot(x='Room_Occupancy_Count', y=feat, data=df, jitter=True, palette='Set2', ax=ax, alpha=0.6)
    sns.boxplot(x='Room_Occupancy_Count', y=feat, data=df, showcaps=False, boxprops={'facecolor':'none'}, showfliers=False, ax=ax)
    ax.set_title(f'{feat} par nombre d\'occupants')
    ax.set_xlabel('Nombre d\'occupants')
    ax.set_ylabel(feat)
    plt.tight_layout()
    save_and_show(fig, f'07_scatter_{feat}_by_target.png')

# 8. Pairplot de toutes les features d'entrée deux à deux (si ≤ 8 features)
if len(features) <= 8:
    print("Pairplot de toutes les features (peut être long)...")
    pairplot = sns.pairplot(df[features], corner=True, plot_kws={'alpha':0.6, 's':20})
    pairplot.fig.suptitle('Pairplot de toutes les features', y=1.02)
    plt.tight_layout()
    pairplot.savefig(os.path.join(OUT_DIR, '08_pairplot_all_features.png'), dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()
else:
    print(f"Trop de features ({len(features)}) pour un pairplot complet. Limité à 8 maximum pour la lisibilité.")

print("Tous les graphiques ont été générés et affichés. Les fichiers PNG sont dans le dossier 'graphes/'.")