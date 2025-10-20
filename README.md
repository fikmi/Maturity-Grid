# Maturity Grid Dashboard

Ce projet fournit un tableau de bord Streamlit permettant de suivre la maturité des équipes selon les axes Conception, Développement, Test et Release. L'application fonctionne uniquement avec des données locales (fichiers ou dossier) afin d'éviter toute dépendance externe.

## Prérequis

- Python 3.10 ou supérieur
- pip

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

Par défaut, l'application propose de charger les fichiers d'exemple présents dans `sample_data/`.

## Fonctionnalités principales

- Chargement multi-fichiers (CSV, JSON, JSON Lines, TXT tabulé ou libre) via upload ou dossier local.
- Normalisation automatique des colonnes (team, axis, metric, status, value, created_at, ended_at, etc.).
- Nettoyage de l'encodage (UTF-8, UTF-8-SIG, Latin-1) et détection des séparateurs (`;`, `,`, `\t`, `|`).
- Calcul d'indicateurs clés par axe : taux de préparation, vélocité, taux de succès des tests, fréquence de release, etc.
- Vue synthétique (scores, tendances, dernier rafraîchissement) et vues détaillées (graphiques et tableau filtrable).
- Export des KPI filtrés en CSV et sauvegarde optionnelle du jeu unifié en CSV ou DuckDB.
- Option d'auto-rafraîchissement configurables (30 à 120 secondes).

## Données d'exemple

Trois fichiers sont fournis dans `sample_data/` :

- `issues.jsonl` : tickets hétérogènes (schéma mixte JSON/JSONL).
- `runs.csv` : exécution de pipelines avec séparateur `;`.
- `events.txt` : extractions tabulaires.

Ces fichiers illustrent le mapping de colonnes et les différents formats supportés.

## Structure du code

Le fichier `app.py` contient l'ensemble du code Streamlit, organisé autour des fonctions suivantes :

- `read_any` : lecture robuste multi-format.
- `detect_separator` : détection heuristique des séparateurs pour CSV/TXT.
- `coerce_datetime` : normalisation des dates.
- `standardize_columns` : mapping des colonnes et harmonisation des axes/statuts.
- `compute_kpis` : calcul des indicateurs clés pour la période filtrée.
- `score_axes` : calcul des scores de maturité (0-5) par axe et score global.
- `make_charts` : préparation des données pour les graphiques et tendances.

## Tests rapides

Pour vérifier la cohérence du code sans lancer l'UI :

```bash
python -m compileall app.py
```

## Licence

Projet fourni à titre d'exemple pédagogique.
