"""
Vérifie le regroupement:
- 2 lignes pour le même patient -> 1 document patient
- ...contenant admissions: [ {...}, {...} ]
"""
import os
import sys
import pandas as pd

# Assure que 'src/' est importable (structure: src/ingest/app.py)
sys.path.insert(0, os.path.abspath("src"))

from ingest.app import build_patient_docs


def test_grouping_two_rows_same_patient(tmp_path, monkeypatch):
    # Mini DataFrame en dur
    df = pd.DataFrame([
        {"patient_id": 1, "gender": "M", "age": 40, "admission_id": 10, "admission_date": "2024-09-01"},
        {"patient_id": 1, "gender": "M", "age": 41, "admission_id": 11, "admission_date": "2024-09-10"},
    ])

    # Config minimale pour le regroupement
    cfg = {
        "patient_key": "patient_id",
        "patient_fields": ["patient_id", "gender", "age"],
        "admission_fields": ["admission_id", "admission_date"],
        # Décommente si ton build_patient_docs caste les dates quand indiqué dans "casts"
        # "casts": {"admission_date": "date"}
    }

    docs = build_patient_docs(df, cfg)

    # => 1 document patient
    assert len(docs) == 1
    doc = docs[0]
    assert doc["patient_id"] == 1

    # => 2 admissions regroupées
    assert "admissions" in doc and isinstance(doc["admissions"], list)
    assert len(doc["admissions"]) == 2
    assert {a["admission_id"] for a in doc["admissions"]} == {10, 11}


