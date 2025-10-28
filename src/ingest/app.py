"""
ETL CSV -> MongoDB (Groupement par patient) + Validation
-------------------------------------------------------
Chaque patient (Name) devient 1 document Mongo
avec ses admissions regroupées dans un tableau.

Ce script :
1️⃣ Charge le CSV
2️⃣ Renomme les colonnes pour uniformiser
3️⃣ Cast les types selon config.yaml
4️⃣ Normalise certains champs (gender, room_number)
5️⃣ Regroupe par patient
6️⃣ Supprime la contrainte "minimum: 0" sur billing_amount (collMod, avec fallback admin)
7️⃣ Insère dans MongoDB (par lots)
8️⃣ Crée/assure les index (robuste)
9️⃣ (NOUVEAU) Valide et écrit ./data/reports/validation_report.json
"""

import os
import glob
import json
import warnings
import argparse
from datetime import datetime, date
from typing import List, Dict, Any

import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import OperationFailure, BulkWriteError
from dotenv import load_dotenv
import yaml
from bson import BSON


# --- Sécurité : rester largement sous la limite BSON 16 MB ---
MAX_BSON_DOC = 15 * 1024 * 1024  # 15MB marge sous la limite 16MB


# ======================================================================
#                         OUTILS GÉNÉRAUX
# ======================================================================
def bson_size(doc: dict) -> int:
    """Taille BSON estimée d'un document."""
    return len(BSON.encode(doc))


def chunk_list(seq, size):
    """Découpe une liste en sous-listes de taille 'size' (itératif)."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def to_bson_datetime(v):
    """
    Convertit date/str/pandas.Timestamp en datetime.datetime (naive, 00:00:00).
    Mongo attend un 'date' BSON = datetime python.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, datetime):
        return v  # déjà bon
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day)  # 00:00:00
    try:
        ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


def env_bool(name: str, default: bool = False) -> bool:
    """Lit une variable d'environnement booléenne."""
    v = os.getenv(name)
    return default if v is None else str(v).lower() in {"1", "true", "yes", "y", "on"}


def load_config(path: str) -> dict:
    """
    Charge config.yaml et remplace ${VAR} par les valeurs d'environnement.
    Permet d'utiliser des env vars dans le YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    for k, v in os.environ.items():
        raw = raw.replace(f"${{{k}}}", v)
    return yaml.safe_load(raw)


def parse_date(val, fmts):
    """Essaie plusieurs formats de date jusqu’à trouver le bon (renvoie date)."""
    if pd.isna(val):
        return None
    for f in fmts:
        try:
            return datetime.strptime(str(val), f).date()
        except Exception:
            pass
    return None


def cast_series(s: pd.Series, target: str, date_fmts: List[str]):
    """
    Convertit les colonnes selon le typage du YAML.
    NB: pour les entiers avec trous, on utilise Int64 (nullable).
    """
    if target == "int":
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if target == "float":
        return pd.to_numeric(s, errors="coerce")
    if target == "bool":
        mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
        return s.astype(str).str.lower().map(mapping)
    if target == "date":
        return s.map(lambda x: parse_date(x, date_fmts))
    if target == "str":
        # Garder tel quel mais en string
        return s.astype(str)
    return s


def _last_non_null(series: pd.Series):
    """Retourne la dernière valeur non vide d'une colonne (utile pour l'entête patient)."""
    for val in reversed(series.tolist()):
        if pd.notna(val) and str(val).strip() != "":
            return val
    return None


# ======================================================================
#                   NORMALISATIONS SPÉCIFIQUES DONNÉES
# ======================================================================
def normalize_gender_series(s: pd.Series) -> pd.Series:
    """
    Normalise 'gender' vers l'enum attendu par le validateur: 'M' | 'F' | 'X' | None.
    Accepte diverses écritures ('male', 'Female', 'other', etc.).
    """
    if s is None:
        return s

    def norm(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        val = str(x).strip().lower()
        if val in {"m", "male", "man", "homme"}:
            return "M"
        if val in {"f", "female", "woman", "femme"}:
            return "F"
        if val in {"x", "other", "non-binary", "non binaire", "autre"}:
            return "X"
        # si inconnu, on met None pour ne pas casser la validation
        return None

    return s.map(norm)


def normalize_room_number_series(s: pd.Series) -> pd.Series:
    """
    Force room_number en numérique nullable (int64 pandas) pour matcher le schema.
    (Le CSV peut contenir '255' en string -> on convertit.)
    """
    if s is None:
        return s
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# ======================================================================
#                     GROUPAGE PAR PATIENT -> DOCUMENTS
# ======================================================================
def build_patient_docs(df: pd.DataFrame, cfg: dict) -> List[Dict[str, Any]]:
    """
    Regroupe les admissions par patient.
    Exemple document final :
    {
      "name": "Jane Doe",
      "age": 35,
      "gender": "F",
      "admissions": [ {...}, {...} ]
    }
    """
    patient_key = cfg.get("patient_key")
    p_fields = cfg.get("patient_fields", [])
    a_fields = cfg.get("admission_fields", [])
    date_fields = {k for k, v in cfg.get("casts", {}).items() if v == "date"}
    docs = []

    for pid, g in df.groupby(patient_key, dropna=False):
        g = g.reset_index(drop=True)

        # --- Partie patient : dernières valeurs non nulles pour chaque champ ---
        patient = {}
        for c in p_fields:
            if c in g.columns:
                patient[c] = _last_non_null(g[c])
        patient[patient_key] = patient.get(patient_key, pid)

        # --- Partie admissions : une admission par ligne ---
        admissions = []
        for _, row in g.iterrows():
            sub = {}
            for c in a_fields:
                if c in g.columns:
                    val = row[c]
                    if c in date_fields:
                        val = to_bson_datetime(val)
                    # normalisation NaN/NA -> None (sauf pour les strings déjà valides)
                    sub[c] = None if (pd.isna(val) if not isinstance(val, str) else False) else val
            admissions.append(sub)

        patient["admissions"] = admissions
        docs.append(patient)

    return docs


# ======================================================================
#           OUTILS VALIDATEUR : suppression du minimum sur billing
# ======================================================================
def _try_get_validator(db, coll_name: str) -> dict:
    """Récupère le validator de la collection (ou {} si absent/inaccessible)."""
    try:
        info = db.command({"listCollections": 1, "filter": {"name": coll_name}})
        batch = info.get("cursor", {}).get("firstBatch", [])
        if batch:
            return batch[0].get("options", {}).get("validator", {}) or {}
    except Exception as e:
        print(f"[WARN] get_validator a échoué: {e}")
    return {}


def _remove_min_and_collmod(db_, coll_name: str, validator_: dict) -> bool:
    """
    Enlève 'minimum: 0' sur admissions[].billing_amount dans validator_,
    puis applique collMod. Renvoie True si modification effectuée.
    """
    changed = False
    try:
        props = validator_["$jsonSchema"]["properties"]
        a_items = props["admissions"]["items"]["properties"]
        billing = a_items.get("billing_amount", {})
        if "minimum" in billing:
            billing.pop("minimum", None)
            changed = True
    except Exception:
        pass

    if not changed:
        return False

    db_.command({"collMod": coll_name, "validator": validator_})
    print("[INFO] Validateur mis à jour (suppression de minimum sur admissions.billing_amount).")
    return True


def ensure_billing_negative_allowed(db, coll_name: str):
    """
    Enlève 'minimum: 0' sur admissions.billing_amount.
    - 1er essai : avec l'utilisateur courant (app).
    - Si échec (droits insuffisants / pas d'accès au validator), on bascule
      automatiquement sur le compte admin fourni par l'env (fallback).
    """
    # --- Tentative avec l'utilisateur applicatif ---
    current = _try_get_validator(db, coll_name)
    if current:
        try:
            if _remove_min_and_collmod(db, coll_name, current):
                return
            else:
                print("[INFO] Validateur inchangé (pas de 'minimum' à enlever côté utilisateur courant).")
                return
        except OperationFailure as e:
            print(f"[WARN] collMod refusé avec l'utilisateur courant ({e}). On tente en admin...)")

    # --- Fallback admin si disponible ---
    root_user = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    root_pwd = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    host = os.getenv("MONGO_HOST", "localhost")
    port = int(os.getenv("MONGO_PORT", "27017"))
    dbname = db.name

    if not (root_user and root_pwd):
        print("[WARN] Pas d'identifiants admin dans l'env; on ne peut pas assouplir le validateur.")
        return

    try:
        admin_client = MongoClient(
            host=host, port=port, username=root_user, password=root_pwd, authSource="admin"
        )
        admin_db = admin_client[dbname]
        val_admin = _try_get_validator(admin_db, coll_name)
        if not val_admin:
            print("[WARN] Impossible de récupérer le validateur en admin (peut-être absent).")
            return
        if not _remove_min_and_collmod(admin_db, coll_name, val_admin):
            print("[INFO] Validateur déjà OK (aucun minimum à enlever) côté admin.")
    except Exception as e:
        print(f"[WARN] Echec de modification du validateur en admin: {e}")


# ======================================================================
#                       OUTIL INDEXATION (robuste)
# ======================================================================
def ensure_indexes(coll, indexes_cfg):
    """
    Crée les index listés dans config.yaml, SANS casser ceux déjà présents.
    - Si un index de même nom existe, on le garde (évite IndexKeySpecsConflict).
    - On ne force PAS 'sparse' ici (risque de conflit avec init.js).
    """
    existing = coll.index_information()  # dict: name -> spec
    for field, order, unique in indexes_cfg:
        direction = ASCENDING if str(order).upper() == "ASC" else DESCENDING
        desired_name = f"{field}_{1 if direction == ASCENDING else -1}"

        if desired_name in existing:
            print(f"[INFO] Index déjà présent, on garde: {desired_name} -> {existing[desired_name]}")
            continue

        try:
            coll.create_index(
                [(field, direction)],
                name=desired_name,
                unique=bool(unique)
            )
            print(f"[INFO] Index créé: {desired_name} (unique={bool(unique)})")
        except OperationFailure as e:
            # 85/86 = conflits de nom/spéc; on log et on continue
            if getattr(e, "code", None) in (85, 86):
                print(f"[WARN] Conflit d'index pour {desired_name} ({e.code}). On ignore: {e}")
                continue
            raise


# ======================================================================
#                           VALIDATION (NOUVEAU)
# ======================================================================
def validate_and_write_report(cfg: dict, client: MongoClient) -> dict:
    """
    Valide la migration et écrit ./data/reports/validation_report.json.
    Renvoie le rapport (dict).
    """
    db = client[cfg["database"]]
    coll = db[cfg["collection"]]

    report = {"status": "running", "message": "", "stats": {}, "checks": []}

    # --- Recompter patients attendus depuis les CSV ---
    csv_glob = os.getenv("CSV_GLOB", "/data/input/*.csv")
    files = sorted(glob.glob(csv_glob))
    if not files:
        report["checks"].append({"name": "csv_found", "ok": False, "info": f"Aucun CSV trouvé à {csv_glob}"})
        expected_patients = None
    else:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        # appliquer rename_map pour retrouver la clé patient
        rename_map = cfg.get("rename_map", {})
        if rename_map:
            df = df.rename(columns=rename_map)
        patient_key = cfg.get("patient_key")
        if patient_key not in df.columns:
            report["checks"].append({"name": "patient_key_in_csv", "ok": False,
                                     "info": f"Colonne '{patient_key}' absente après rename_map"})
            expected_patients = None
        else:
            expected_patients = df[patient_key].dropna().astype(str).str.strip().nunique()
            report["stats"]["expected_distinct_patients_from_csv"] = expected_patients
            report["checks"].append({"name": "csv_loaded", "ok": True,
                                     "info": f"{len(df)} lignes, {expected_patients} patients distincts"})

    # --- Comptage côté Mongo ---
    actual_docs = coll.count_documents({})
    report["stats"]["mongo_documents"] = actual_docs
    report["checks"].append({"name": "mongo_count", "ok": True,
                             "info": f"{actual_docs} documents dans {cfg['collection']}"})

    if expected_patients is not None:
        ok_cnt = (expected_patients == actual_docs)
        report["checks"].append({
            "name": "count_match",
            "ok": ok_cnt,
            "info": f"CSV patients={expected_patients} vs Mongo docs={actual_docs}"
        })

    # --- Vérifier index attendus ---
    idx_info = coll.index_information()  # name -> spec
    expected_indexes = cfg.get("indexes", [])
    missing = []
    for field, order, unique in expected_indexes:
        direction = ASCENDING if str(order).upper() == "ASC" else DESCENDING
        name = f"{field}_{1 if direction == ASCENDING else -1}"
        if name not in idx_info:
            missing.append(name)
    report["checks"].append({
        "name": "indexes_present",
        "ok": len(missing) == 0,
        "info": "OK" if not missing else f"Manquants: {', '.join(missing)}"
    })

    # --- Échantillon de documents: structure & types de base ---
    sample = list(coll.find({}, {"_id": 0}).limit(200))
    basic_ok = True
    issues = []
    gender_ok_values = {"M", "F", "X", None}

    p_fields = set(cfg.get("patient_fields", []))
    a_fields = cfg.get("admission_fields", [])
    date_fields = {k for k, v in cfg.get("casts", {}).items() if v == "date"}

    def is_datetime_or_none(v):
        if v is None:
            return True
        return hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day")

    for i, doc in enumerate(sample):
        # admissions présent et list
        if "admissions" not in doc or not isinstance(doc["admissions"], list):
            basic_ok = False
            issues.append(f"doc#{i}: 'admissions' absent ou non-list")
            continue
        # champs patient présents (si définis)
        for pf in p_fields:
            if pf not in doc:
                basic_ok = False
                issues.append(f"doc#{i}: champ patient manquant '{pf}'")
        # validations admissions
        for j, adm in enumerate(doc["admissions"][:5]):  # 5 premières admissions de chaque doc
            if not isinstance(adm, dict):
                basic_ok = False
                issues.append(f"doc#{i} adm#{j}: non-dict")
                continue
            for af in a_fields:
                if af not in adm:
                    basic_ok = False
                    issues.append(f"doc#{i} adm#{j}: champ admission manquant '{af}'")
            # dates
            for dfld in date_fields:
                if dfld in adm and not is_datetime_or_none(adm[dfld]):
                    basic_ok = False
                    issues.append(f"doc#{i} adm#{j}: '{dfld}' n'est pas date/None")
        # gender enum
        if "gender" in doc:
            val = doc["gender"]
            if val not in gender_ok_values:
                basic_ok = False
                issues.append(f"doc#{i}: gender invalide '{val}'")

    report["checks"].append({
        "name": "sample_structure_validation",
        "ok": basic_ok,
        "info": "OK" if basic_ok else f"Problèmes: {len(issues)} (voir 'problems')"
    })
    if issues:
        report["problems"] = issues[:200]  # limite

    # --- Statut global ---
    all_ok = all(c.get("ok", False) for c in report["checks"] if c["name"] != "count_match") and \
             all(c.get("ok", True) for c in report["checks"] if c["name"] == "count_match")
    report["status"] = "ok" if all_ok else "fail"
    report["message"] = "Validation terminée"

    # --- Écriture du rapport ---
    out_dir = "./data/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"[INFO] Rapport écrit: {out_path} (status={report['status']})")
    return report


# ======================================================================
#                                MAIN
# ======================================================================
def run_migration(cfg: dict):
    csv_glob = os.getenv("CSV_GLOB", "/data/input/*.csv")
    strict_types = env_bool("STRICT_TYPES", True)

    files = sorted(glob.glob(csv_glob))
    if not files:
        print(f"[WARN] No CSV found at {csv_glob}")
        return 0

    # Lecture de tous les CSV concaténés
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # --- Renommer les colonnes selon rename_map ---
    rename_map = cfg.get("rename_map", {})
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[INFO] Colonnes renommées : {rename_map}")

    # --- Casting des types selon YAML ---
    for col, target in cfg.get("casts", {}).items():
        if col in df.columns:
            df[col] = cast_series(df[col], target, cfg.get("date_formats", []))
        elif strict_types:
            warnings.warn(f"Missing column for cast: {col}")

    # --- Normalisations spécifiques pour coller au validateur Mongo ---
    # 1) gender -> 'M'|'F'|'X'|None
    if "gender" in df.columns:
        df["gender"] = normalize_gender_series(df["gender"])

    # 2) room_number -> entier nullable
    if "room_number" in df.columns:
        df["room_number"] = normalize_room_number_series(df["room_number"])

    # --- Groupement par patient -> documents prêts pour Mongo ---
    docs = build_patient_docs(df, cfg)

    # --- Connexion à MongoDB (utilisateur applicatif) ---
    client = MongoClient(
        host=os.getenv("MONGO_HOST"),
        port=int(os.getenv("MONGO_PORT", "27017")),
        username=os.getenv("MONGO_APP_USERNAME"),
        password=os.getenv("MONGO_APP_PASSWORD"),
        authSource=os.getenv("MONGO_DB"),
    )
    db = client[cfg["database"]]
    coll = db[cfg["collection"]]

    # --- IMPORTANT : desserrer le validateur pour billing_amount (avec fallback admin) ---
    ensure_billing_negative_allowed(db, cfg["collection"])

    # --- Insertion par lots (messages < 16MB) ---
    inserted = 0
    if not docs:
        print("[INFO] No documents to insert.")
    else:
        BATCH_SIZE = 1000  # Sûr. On peut augmenter si besoin.
        try:
            for i in range(0, len(docs), BATCH_SIZE):
                batch = docs[i:i + BATCH_SIZE]
                res = coll.insert_many(batch, ordered=False)
                inserted += len(res.inserted_ids)
            print(f"[INFO] Inserted {inserted} patient documents in batches of {BATCH_SIZE}.")
        except BulkWriteError as bwe:
            details = bwe.details or {}
            write_errors = details.get("writeErrors", [])
            print(f"[ERROR] BulkWriteError: {len(write_errors)} erreurs. Détails (extrait):")
            for e in write_errors[:5]:
                print(f"  - idx={e.get('index')} code={e.get('code')} err={e.get('errmsg')}")
            raise

    # --- Indexes robustes (ne casse pas les index existants, évite les conflits) ---
    ensure_indexes(coll, cfg.get("indexes", []))
    print("[INFO] Indexes ensured.")
    print("[INFO] Migration completed successfully.")
    return inserted


def build_mongo_client(cfg: dict) -> MongoClient:
    return MongoClient(
        host=os.getenv("MONGO_HOST"),
        port=int(os.getenv("MONGO_PORT", "27017")),
        username=os.getenv("MONGO_APP_USERNAME"),
        password=os.getenv("MONGO_APP_PASSWORD"),
        authSource=os.getenv("MONGO_DB"),
    )


def main():
    load_dotenv(override=True)
    cfg = load_config("config.yaml")

    parser = argparse.ArgumentParser(description="ETL CSV -> MongoDB + Validation")
    sub = parser.add_subparsers(dest="command")

    p_migrate = sub.add_parser("migrate", help="Exécuter l'ETL (par défaut)")
    p_migrate.add_argument("--and-validate", action="store_true",
                           help="Exécuter l'ETL puis la validation")

    sub.add_parser("validate", help="Exécuter uniquement la validation")

    # Comportement par défaut = migrate
    args = parser.parse_args()
    if args.command in (None, "migrate"):
        inserted = run_migration(cfg)
        if args.command == "migrate" and getattr(args, "and-validate", False):
            client = build_mongo_client(cfg)
            validate_and_write_report(cfg, client)
    elif args.command == "validate":
        client = build_mongo_client(cfg)
        validate_and_write_report(cfg, client)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
