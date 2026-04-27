# 🏥 Prédiction des Charges Médicales
### Projet Machine Learning — Régression 

---
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-FF9F00?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)

---
## 📋 Table des Matières

1. [Contexte](#contexte)
2. [Objectif](#objectif)
3. [Structure du Projet](#structure-du-projet)
4. [Dataset](#dataset)
5. [Installation & Prérequis](#installation--prérequis)
6. [Utilisation](#utilisation)
7. [Méthodologie](#méthodologie)
8. [Conclusion](#conclusion)

---

## Contexte

Ce projet s'inscrit dans le cadre du module **Machine Learning / Régression**.  
Il porte sur l'analyse et la modélisation d'un jeu de données de **charges médicales individuelles** aux États-Unis.

L'objectif est d'appliquer une démarche complète de Data Science :  
**Exploration → Prétraitement → Modélisation → Évaluation → Sélection de Features**

---

## Objectif

> **Prédire le montant des charges médicales (`charges`)** facturées à un individu en fonction de ses caractéristiques personnelles.

---

## Structure du Projet

```
📦 Machine_Learning-charges_medical/
├── 📋 cahier_de_charges.md                  # Spécifications du projet
├── 📋 cahier_de_charges.pdf                 # Spécifications du projet
├── 📓 prediction_charges_medicales.ipynb   # Notebook principal (exécuté)
├── 📊 medical-charges.csv                  # Dataset source
└── 📄 README.md                            # Ce fichier
```

---

## Dataset

**Fichier :** `medical-charges.csv`  
**Observations :** 1 338 (après suppression d'1 doublon)  
**Variables :** 7 (6 features + 1 target)

| Variable | Type | Description | Valeurs |
|---|---|---|---|
| `age` | Numérique | Âge du bénéficiaire | 18 – 64 ans |
| `sex` | Catégorielle | Sexe | `female`, `male` |
| `bmi` | Numérique | Indice de Masse Corporelle | 15.96 – 53.13 |
| `children` | Numérique | Nombre d'enfants couverts | 0 – 5 |
| `smoker` | Catégorielle | Statut fumeur | `yes`, `no` |
| `region` | Catégorielle | Zone géographique US | `southwest`, `southeast`, `northwest`, `northeast` |
| `charges` | **Numérique** | **Charges médicales (cible)** | 1 121 – 63 770 $ |

---

## Installation & Prérequis

### Prérequis
- Python 3.8+
- Jupyter Notebook ou JupyterLab

### Installation des dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
```


### Lancement du Notebook

```bash
jupyter notebook prediction_charges_medicales.ipynb
```

---

## Utilisation

1. Cloner ou télécharger le projet
2. Placer `medical-charges.csv` dans le même dossier que le notebook
3. Ouvrir `prediction_charges_medicales.ipynb` dans Jupyter
4. Exécuter toutes les cellules : **Kernel → Restart & Run All**

---

## Méthodologie

Le projet suit **7 phases** :

### Phase 1 — Analyse Exploratoire (EDA)
- Inspection initiale : `shape`, `info`, `describe`
- Analyse univariée : histogrammes, boxplots, countplots
- Analyse bivariée : scatter plots, boxplots par catégorie
- Impact du statut `smoker` sur les `charges`
- Interaction `bmi × smoker`
- Matrice de corrélation + pairplot
- Détection des outliers par méthode IQR

### Phase 2 — Nettoyage & Prétraitement
- Vérification valeurs manquantes → **0 NaN**
- Suppression doublons → **1 doublon supprimé**
- Encodage :
  - `sex`, `smoker` → **Label Encoding** (variables binaires)
  - `region` → **One-Hot Encoding** (4 modalités sans ordre)
- Scaling → **StandardScaler** sur `age`, `bmi`, `children`
- Split → **80% train / 20% test** (`random_state=42`)

### Phase 3 — Modélisation

| # | Modèle | Régularisation | Alpha optimal |
|---|---|---|---|
| 1 | Linear Regression | Aucune (OLS) | — |
| 2 | Ridge | L2 | **2.1049** |
| 3 | Lasso | L1 | **28.4804** |
| 4 | ElasticNet | L1 + L2 | **28.4804** / l1_ratio=0.5 |

> Hyperparamètres optimisés via **RidgeCV** et **LassoCV** (5-fold cross-validation).

### Phase 4 — Évaluation des Modèles
- Métriques : **RMSE**, **MAE**, **R²**, **R² Ajusté**, **MAPE**
- Analyse des résidus (distribution + scatter résidus vs prédictions)
- Visualisation réelles vs prédites

### Phase 5 — Sélection de Features
- **Backward Elimination** (statsmodels OLS, seuil p-value = 0.05)
- **RFE** — Recursive Feature Elimination (top 5 features)
- **Analyse Lasso** — coefficients mis à zéro

### Phase 6 — Comparaison All Features vs Selected Features
- Ré-entraînement des 4 modèles avec les features sélectionnées
- Tableau comparatif et bar chart groupé

### Phase 7 — Conclusion
- Synthèse des résultats
- Observations clés
- Limites et perspectives

---

## Conclusion

Les modèles linéaires atteignent d’excellentes performances avec toutes les features. La sélection de features par Backward Elimination permet de simplifier le modèle sans perte significative de performance.

### Pistes d'amélioration
- **Feature Engineering** : variable d'interaction `bmi × smoker`, transformation `log(charges)`
- **Modèles non-linéaires** : Random Forest, XGBoost, LightGBM (cible R² > 0.85)
- **Données supplémentaires** : antécédents médicaux, activité physique, historique sinistres

---

## Outils & Technologies

| Catégorie | Outils |
|---|---|
| Langage | Python 3.x |
| Environnement | Jupyter Notebook |
| Librairies | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels` |
| IDE | VS Code / JupyterLab |

---

## 👤 Réalisé par 

BOUAZIZ Ameni 
<br>
TRABELSI Asma
