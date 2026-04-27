# Cahier de Charges — Prédiction des Charges Médicales

## 1. Contexte et Objectif du Projet

### 1.1 Contexte
Ce projet s'inscrit dans le cadre du module **Machine Learning / Régression** du Master ISI. Il porte sur l'analyse et la modélisation d'un jeu de données de charges médicales individuelles aux États-Unis. L'objectif est d'appliquer une démarche complète de Data Science : de l'exploration des données jusqu'à la modélisation et l'évaluation des performances.

### 1.2 Objectif Principal
**Prédire le montant des charges médicales (`charges`)** facturées à un individu en fonction de ses caractéristiques personnelles (âge, sexe, IMC, nombre d'enfants, statut fumeur, région géographique).

### 1.3 Jeu de Données
- **Fichier** : `medical-charges.csv`
- **Nombre d'observations** : 1 339
- **Nombre de variables** : 7 (6 features + 1 target)

---

## 2. Description du Dataset

| Variable   | Type         | Description                                                    | Valeurs possibles                              |
|------------|--------------|----------------------------------------------------------------|------------------------------------------------|
| `age`      | Numérique    | Âge du bénéficiaire principal                                  | 18 – 64 ans                                    |
| `sex`      | Catégorielle | Sexe du bénéficiaire                                           | `female`, `male`                               |
| `bmi`      | Numérique    | Indice de Masse Corporelle (Body Mass Index)                   | Valeur continue (ex. 15.96 – 53.13)            |
| `children` | Numérique    | Nombre d'enfants / dépendants couverts par l'assurance         | 0 – 5                                          |
| `smoker`   | Catégorielle | Le bénéficiaire est-il fumeur ?                                | `yes`, `no`                                    |
| `region`   | Catégorielle | Zone résidentielle du bénéficiaire aux États-Unis              | `southwest`, `southeast`, `northwest`, `northeast` |
| `charges`  | Numérique    | Charges médicales individuelles facturées par l'assurance (**variable cible**) | Valeur continue (ex. 1 121 – 63 770)          |

---

## 3. Travail Demandé

Le projet se décompose en **7 phases** détaillées ci-dessous :

---

### Phase 1 — Analyse Exploratoire des Données (EDA)

L'exploration doit fournir une compréhension approfondie du dataset avant toute transformation.

#### Tâches :
1. **Chargement et inspection initiale** :
   - `df.head()`, `df.tail()`, `df.shape`, `df.info()`, `df.describe()`
   - Vérification des types de données
2. **Analyse univariée** :
   - Distribution de chaque variable numérique (`age`, `bmi`, `children`, `charges`) via histogrammes et boxplots
   - Comptage des modalités pour les variables catégorielles (`sex`, `smoker`, `region`) via countplots
3. **Analyse bivariée** :
   - Relation entre chaque feature et la variable `charges` (scatter plots, boxplots par catégorie)
   - Analyse de l'impact du statut `smoker` sur les `charges`
   - Analyse de l'interaction `bmi × smoker` sur les `charges`
4. **Analyse multivariée** :
   - Matrice de corrélation (heatmap) entre les variables numériques
   - Pairplot coloré par `smoker`
5. **Détection des outliers** :
   - Méthode IQR (Interquartile Range) sur les variables numériques
   - Visualisation via boxplots

#### Livrables :
- Visualisations commentées (matplotlib / seaborn)
- Observations et conclusions textuelles dans des cellules Markdown

---

### Phase 2 — Nettoyage et Prétraitement (Cleaning & Preprocessing)

#### Tâches :
1. **Gestion des valeurs manquantes** :
   - Vérification avec `df.isnull().sum()`
   - Stratégie de traitement si nécessaire (imputation ou suppression)
2. **Gestion des doublons** :
   - Détection avec `df.duplicated().sum()`
   - Suppression si doublons exacts
3. **Encodage des variables catégorielles** :
   - **Label Encoding** ou **One-Hot Encoding** pour `sex`, `smoker`, `region`
   - Justification du choix d'encodage
4. **Feature Scaling** :
   - Standardisation (`StandardScaler`) ou Normalisation (`MinMaxScaler`)
   - Application sur les variables numériques
5. **Séparation des données** :
   - Définition de `X` (features) et `y` (target = `charges`)
   - Split `train / test` (80% / 20%) avec `train_test_split` et `random_state` fixé

#### Livrables :
- Dataset nettoyé et transformé prêt pour la modélisation
- Justification de chaque choix de prétraitement

---

### Phase 3 — Modélisation (Linear Regression + Régularisation)

#### Modèles à implémenter :
| # | Modèle              | Description                                                    |
|---|----------------------|----------------------------------------------------------------|
| 1 | **Linear Regression** | Régression linéaire simple (OLS – Ordinary Least Squares)      |
| 2 | **Ridge Regression**  | Régularisation L2 — pénalise les grands coefficients (α ≠ 0)  |
| 3 | **Lasso Regression**  | Régularisation L1 — peut mettre certains coefficients à 0     |
| 4 | **ElasticNet**        | Combinaison de L1 + L2 (paramètres α et l1_ratio)             |

#### Tâches :
1. Entraîner chaque modèle sur le jeu d'entraînement
2. Effectuer des prédictions sur le jeu de test
3. Afficher et interpréter les **coefficients** de chaque modèle
4. Utiliser `cross_validation` pour évaluer la robustesse
5. Optionnel : recherche des meilleurs hyperparamètres via `GridSearchCV` ou `RidgeCV` / `LassoCV`

#### Livrables :
- Modèles entraînés et prédictions
- Tableau comparatif des coefficients

---

### Phase 4 — Évaluation des Modèles

#### Métriques d'évaluation :
| Métrique            | Formule / Description                                                  |
|---------------------|------------------------------------------------------------------------|
| **RMSE**            | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ — Erreur quadratique moyenne |
| **MAE**             | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ — Erreur absolue moyenne              |
| **R²**              | Coefficient de détermination — proportion de variance expliquée        |
| **Adjusted R²**     | R² ajusté tenant compte du nombre de features                         |
| **MAPE** (optionnel)| Erreur absolue moyenne en pourcentage                                  |

#### Tâches :
1. Calculer les métriques ci-dessus pour **chaque modèle**
2. Créer un **tableau récapitulatif** comparant les performances
3. Visualiser les **résidus** (distribution, scatter plot résidus vs prédictions)
4. Tracer les **valeurs réelles vs prédites** (scatter plot)
5. Interpréter les résultats et identifier le meilleur modèle

#### Livrables :
- Tableau comparatif des métriques (DataFrame)
- Graphiques d'analyse des résidus
- Interprétation textuelle

---

### Phase 5 — Sélection de Features (Feature Selection)

#### Méthodes à appliquer :
1. **Backward Elimination** :
   - Partir de toutes les features
   - Supprimer itérativement la feature avec la plus grande p-value (seuil > 0.05)
   - Utiliser `statsmodels` OLS pour obtenir les p-values
   - Documenter chaque itération
2. **RFE (Recursive Feature Elimination)** (optionnel) :
   - Utiliser `RFE` de scikit-learn avec `LinearRegression` comme estimateur
   - Sélectionner le nombre optimal de features
3. **Analyse des coefficients Lasso** :
   - Identifier les features mises à zéro par Lasso (coefficient = 0)

#### Tâches :
1. Appliquer la Backward Elimination pas à pas
2. Lister les features éliminées et celles retenues
3. Justifier la sélection finale

#### Livrables :
- Tableau des p-values à chaque itération
- Liste des features sélectionnées
- Explication de l'algorithme en Markdown

---

### Phase 6 — Comparaison : Tous les Features vs Features Sélectionnées

#### Tâches :
1. **Ré-entraîner** tous les modèles (Linear Regression, Ridge, Lasso, ElasticNet) avec les **features sélectionnées** uniquement
2. **Recalculer** toutes les métriques d'évaluation (RMSE, MAE, R², Adj. R²)
3. **Construire un tableau comparatif** :
   - Lignes : modèles
   - Colonnes : métriques avec **All Features** vs **Selected Features**
4. **Visualiser** la comparaison (bar chart groupé)
5. **Analyser** l'impact de la sélection de features :
   - Y a-t-il une amélioration ou dégradation ?
   - Le modèle est-il plus simple sans perte significative de performance ?

#### Livrables :
- Tableau comparatif complet
- Graphiques de comparaison
- Analyse et discussion

---

### Phase 7 — Conclusion

#### Contenu attendu :
1. **Résumé des résultats** :
   - Meilleur modèle identifié
   - Features les plus importantes pour la prédiction
2. **Observations clés** :
   - Impact du statut fumeur sur les charges
   - Rôle de l'âge et de l'IMC
3. **Limites du projet** :
   - Taille du dataset
   - Hypothèses de linéarité
   - Variables manquantes potentielles
4. **Perspectives d'amélioration** :
   - Modèles non-linéaires (Decision Tree, Random Forest, XGBoost)
   - Feature Engineering (interactions, transformations polynomiales)
   - Collecte de données supplémentaires

---

## 4. Outils et Technologies

| Catégorie       | Outils                                          |
|-----------------|------------------------------------------------|
| **Langage**     | Python 3.x                                     |
| **Environnement** | Jupyter Notebook                              |
| **Librairies**  | pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels |
| **IDE**         | VS Code / Jupyter Lab                          |

---

## 5. Structure du Notebook

Le notebook final doit suivre la structure suivante :

```
1. Introduction et Description du Dataset
2. Importation des Librairies
3. Chargement des Données
4. Analyse Exploratoire (EDA)
   4.1 Analyse Univariée
   4.2 Analyse Bivariée
   4.3 Analyse Multivariée
   4.4 Détection des Outliers
5. Nettoyage et Prétraitement
   5.1 Valeurs Manquantes
   5.2 Doublons
   5.3 Encodage
   5.4 Feature Scaling
   5.5 Train/Test Split
6. Modélisation
   6.1 Linear Regression
   6.2 Ridge Regression
   6.3 Lasso Regression
   6.4 ElasticNet Regression
7. Évaluation des Modèles
   7.1 Métriques
   7.2 Analyse des Résidus
   7.3 Tableau Comparatif
8. Sélection de Features
   8.1 Backward Elimination
   8.2 RFE (optionnel)
   8.3 Analyse Lasso
9. Comparaison All Features vs Selected Features
10. Conclusion
```

---

## 6. Critères de Réussite

- [ ] Dataset correctement chargé et exploré
- [ ] EDA complète avec visualisations pertinentes
- [ ] Prétraitement justifié et documenté
- [ ] Au moins 4 modèles de régression entraînés
- [ ] Métriques d'évaluation calculées et comparées
- [ ] Feature Selection appliquée et documentée
- [ ] Comparaison All Features vs Selected Features réalisée
- [ ] Conclusion synthétique et perspectives

---

> **Note** : Chaque phase doit être documentée avec des cellules Markdown explicatives dans le notebook. Le code doit être commenté et reproductible.
