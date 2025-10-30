# 🧬 MED Mongo Migration

## 📘 Description

Ce projet a pour objectif de **migrer des données médicales au format CSV vers une base de données MongoDB**, en respectant un **schéma JSON strict** et en garantissant la qualité et la cohérence des données importées.

L’application :

- charge les fichiers CSV depuis `data/input/`
- nettoie et transforme les données (types, normalisation, regroupement des admissions)
- applique un schéma de validation MongoDB
- insère les documents patients avec leurs admissions
- génère un rapport de validation (`data/reports/validation_report.json`)
- et crée des index pour accélérer les requêtes

Le tout fonctionne entièrement avec **Docker Compose**.

---

## 📁 Structure et architecture du projet

```
med-mongo-migration/
│
├── data/
│   ├── input/                        # Contient les fichiers CSV source
│   └── reports/                      # Contient les rapports de validation
│
├── src/
│   └── ingest/
│       └── app.py                    # Script principal de migration
│
├── tests/
│   ├── test_grouping.py              # Test du regroupement patient/admissions
│   └── test_validation.py            # Tests de validation des données
|
|── docs/
|   └── architecture/
│       └── MED_MONGO_MIGATION.drawio # Schéma architectural
│
├── .env                              # Variables d'environnement (MongoDB & configuration)
├── docker-compose.yml                # Configuration des services Docker
├── Dockerfile                        # Image de migration
└── README.md                         # Ce fichier
```

![MED Mongo Migration Architecture](docs/architecture/med_mongo_architecture.png)
---

## ⚙️ Configuration de l’environnement

### 🧾 Fichier `.env`

```bash
# MongoDB configuration
MONGO_INITDB_ROOT_USERNAME=**********
MONGO_INITDB_ROOT_PASSWORD=**********
MONGO_INITDB_DATABASE=**********

MONGO_APP_USERNAME=**********
MONGO_APP_PASSWORD=**********

# Le service mongo s'appelle "mongo" dans docker compose
MONGO_HOST=**********
MONGO_PORT=**********

MONGO_DB=**********
MONGO_COLLECTION=**********

CSV_GLOB=**********
STRICT_TYPES=**********
```

---

## 🐳 Lancer l’application avec Docker

Assure-toi d’avoir **Docker** et **Docker Compose** installés.

### 1️⃣ Construire les images et lancer les conteneurs

```bash
docker compose up -d
```

### 2️⃣ Lancer la migration

```bash
docker compose run --rm migration
```

### 3️⃣ Une fois la migration terminée

- Les données sont insérées dans MongoDB.
- Le rapport de validation est généré dans :

```
data/reports/validation_report.json
```

### 4️⃣ Vérifier les données dans MongoDB

```bash
docker compose exec mongo mongosh -u ingestor -p "ingestor_password" --authenticationDatabase med_db
> use med_db
> db.patients.count()
```

---

## 🧠 Visualisation dans MongoDB Compass

Pour explorer la base depuis **MongoDB Compass** :

1. Ouvre Compass  
2. Clique sur **New Connection**  
3. Entre l’URI suivante :

```
mongodb://admin:admin_password@localhost:27017/?authSource=admin
```

Tu devrais voir la base **`med_db`** et la collection **`patients`**.

---

## 🧪 Tests unitaires

Les tests vérifient :

- le **regroupement** des admissions pour un même patient  
- la **présence et la structure** des colonnes CSV  
- l’existence du **rapport de validation** après migration  

### ▶️ Exécution des tests

Depuis la racine du projet :

```bash
pytest -v
```

### Exemple de test (`tests/test_grouping.py`)

```python
"""
Vérifie le regroupement:
- 2 lignes pour le même patient -> 1 document patient
- ...contenant admissions: [ {...}, {...} ]
"""
import os
import sys
import pandas as pd

# S'assurer que "src" est dans le PYTHONPATH
sys.path.insert(0, os.path.abspath("src"))

from ingest.app import build_patient_docs

def test_grouping_two_rows_same_patient():
    df = pd.DataFrame([
        {"patient_id": 1, "gender": "M", "age": 40, "admission_id": 10, "admission_date": "2024-09-01"},
        {"patient_id": 1, "gender": "M", "age": 41, "admission_id": 11, "admission_date": "2024-09-10"},
    ])
    cfg = {
        "patient_key": "patient_id",
        "patient_fields": ["patient_id", "gender", "age"],
        "admission_fields": ["admission_id", "admission_date"]
    }
    docs = build_patient_docs(df, cfg)
    assert len(docs) == 1
    doc = docs[0]
    assert doc["patient_id"] == 1
    assert len(doc["admissions"]) == 2
```

---

## 🧾 Rapport de validation

Après chaque migration, un fichier est généré :

```
data/reports/validation_report.json
```

### Exemple

```json
{
  "inserted_documents": 49992,
  "invalid_rows": [],
  "skipped": 8,
  "timestamp": "2025-10-27T10:18:46Z"
}
```

---

## 🧱 Schéma MongoDB

Le validateur JSON impose :

- `name` obligatoire (string)  
- `admissions` (array d’objets)  
- champs numériques (`age`, `room_number`, `billing_amount`) correctement typés  
- `gender` limité à `["M", "F", "X", null]`

Des index sont également créés :

- `name` (unique, sparse)  
- `admissions.date_of_admission`  
- `admissions.hospital`  
- `admissions.doctor`

---

## 🔎 Exemples de requêtes utiles

- **Patients par hôpital**
  ```js
  db.patients.aggregate([
    { $unwind: "$admissions" },
    { $group: { _id: "$admissions.hospital", patients: { $addToSet: "$name" }, count: { $sum: 1 } } },
    { $sort: { count: -1 } }
  ])
  ```

- **Admissions avec factures négatives (crédits/remises)**
  ```js
  db.patients.aggregate([
    { $unwind: "$admissions" },
    { $match: { "admissions.billing_amount": { $lt: 0 } } },
    { $project: { name: 1, "admissions.billing_amount": 1, "admissions.date_of_admission": 1 } },
    { $limit: 50 }
  ])
  ```

- **Dernière admission par patient**
  ```js
  db.patients.aggregate([
    { $project: {
        name: 1,
        lastAdmission: { $max: "$admissions.date_of_admission" }
    }},
    { $sort: { lastAdmission: -1 } }
  ])
  ```

---

## 🧩 Dépannage rapide

- **`authentication failed` dans Compass**  
  Vérifie l’URI et le `authSource=admin`.  
  Utilise : `mongodb://admin:admin_password@localhost:27017/?authSource=admin`

- **Tests qui échouent pour `healthcare_dataset.csv` introuvable**  
  Place un CSV dans `data/input/healthcare_dataset.csv` (ou adapte le chemin dans les tests).

- **Conflits d’index**  
  Les index sont créés de façon idempotente. En cas d’index déjà présent, c’est loggé et ignoré.

---

## 📝 Licence

Projet pédagogique — usage libre dans le cadre de la formation OpenClassrooms"# Projet_5_migration" 
