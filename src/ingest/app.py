# -*- coding: utf-8 -*-
"""
🩺 app.py — Migration CSV vers MongoDB avec nettoyage et regroupement

But du script :
---------------
➡ Lire un fichier CSV contenant les admissions de patients.
➡ Nettoyer et normaliser les données (corriger les majuscules, espaces, etc.).
➡ Regrouper toutes les admissions d’un même patient dans une seule fiche.
➡ Sauvegarder le résultat dans un fichier JSON.
➡ (Optionnel) Insérer ou mettre à jour les patients dans MongoDB.

Le résultat est donc un JSON et une base Mongo avec une structure :
{
  "Name": "Ashley Garcia",
  "Age": 59,
  "Gender": "Male",
  "Blood Type": "O-",
  "Admissions": [ {...}, {...} ]
}
"""

# =======================================================================
# 🧩 IMPORTS : bibliothèques nécessaires
# =======================================================================

import os          # pour lire les variables d’environnement (ex: MONGO_URI)
import json        # pour exporter les données en JSON
import time        # pour les pauses lors de la connexion Mongo
from pathlib import Path   # pour manipuler les chemins de fichiers facilement
from typing import List, Dict, Any   # pour mieux typer les fonctions (lisibilité)

import pandas as pd   # bibliothèque de manipulation de données tabulaires (CSV)
from pymongo import MongoClient, UpdateOne   # client MongoDB et opérations bulk
from pymongo.errors import ServerSelectionTimeoutError, BulkWriteError   # erreurs Mongo


# =======================================================================
# ⚙️ CONFIGURATION & OUTILS DE BASE
# =======================================================================

def load_config():
    """
    Lis les variables d’environnement (ou met des valeurs par défaut)
    Ces variables sont définies dans docker-compose.yml.
    """
    CSV_PATH = os.getenv("CSV_PATH", "/data/input/patients.csv")  # chemin du CSV
    REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/data/reports/migration"))  # dossier où écrire le JSON
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@mongo:27017/?authSource=admin")
    DB_NAME = os.getenv("MONGO_DB") or os.getenv("MONGO_INITDB_DATABASE", "meddb")
    ENABLE_MONGO_INSERT = os.getenv("ENABLE_MONGO_INSERT", "1") == "1"  # activer l’insertion Mongo ?
    return CSV_PATH, REPORTS_DIR, MONGO_URI, DB_NAME, ENABLE_MONGO_INSERT


def get_client(mongo_uri: str) -> MongoClient:
    """Crée une connexion MongoDB (client)."""
    return MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)


def wait_for_mongo(client: MongoClient, retries: int = 30, delay: int = 2) -> None:
    """
    Attends que Mongo soit prêt avant d’insérer.
    - Essaie un 'ping' jusqu’à 30 fois (toutes les 2 secondes).
    """
    for _ in range(retries):
        try:
            client.admin.command("ping")  # si Mongo répond, c’est prêt
            print("✅ MongoDB connecté")
            return
        except ServerSelectionTimeoutError:
            print(f"⏳ MongoDB non prêt, retry dans {delay}s...")
            time.sleep(delay)
    raise RuntimeError("❌ Impossible de se connecter à MongoDB")


# =======================================================================
# 🧹 LECTURE ET RÉSUMÉ DU FICHIER CSV
# =======================================================================

def read_data(path: str) -> pd.DataFrame:
    """Lit un fichier CSV ou JSON et renvoie un DataFrame pandas."""
    print(f"📄 Lecture: {path}")
    if path.lower().endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path, low_memory=False)

def summarize(df: pd.DataFrame) -> None:
    """Affiche des infos utiles sur le jeu de données."""
    print("\nℹ️  Infos dataset:")
    df.info()
    print("\n🔎 Valeurs manquantes:\n", df.isnull().sum())
    print("\n🧮 Doublons (lignes strictes):", df.duplicated().sum())


# =======================================================================
# 🧠 NORMALISATION DES CHAMPS
# =======================================================================

def _clean_str(x: str) -> str:
    """Nettoie les chaînes : supprime les espaces, tabulations, doublons, etc."""
    if pd.isna(x):
        return ""
    s = " ".join(str(x).strip().replace("\t", " ").split())
    s = s.replace(" ,", ",").replace(", ,", ",").strip()
    return s

def normalize_name(raw: str) -> str:
    """Met les noms en 'Title Case' (Ashley Garcia, pas ASHLEY GARCIA)"""
    s = _clean_str(raw)
    if not s:
        return ""
    parts = [p for p in s.replace(",", " ").split(" ") if p]
    parts = [p.lower().capitalize() if len(p) > 1 else p.upper() for p in parts]
    # Correction spéciale pour les noms commençant par "Mc"
    fixed = []
    for p in parts:
        if p.startswith("Mc") and len(p) > 2:
            p = "Mc" + p[2:].capitalize()
        fixed.append(p)
    return " ".join(fixed)

def normalize_gender(raw: str) -> str:
    """Uniformise les genres ('Male' ou 'Female')."""
    s = _clean_str(raw).lower()
    if s in ("male", "m", "masculin"):
        return "Male"
    if s in ("female", "f", "feminin", "féminin"):
        return "Female"
    return s.capitalize() if s else ""

def normalize_blood(raw: str) -> str:
    """Met les groupes sanguins en forme (A+, B-, O-, etc.)"""
    s = _clean_str(raw).upper().replace(" ", "")
    valid = {"A","A+","A-","B","B+","B-","O","O+","O-","AB","AB+","AB-"}
    return s if s in valid else s


# =======================================================================
# 🔄 TRANSFORMATION DES DONNÉES
# =======================================================================

GROUP_COLS_NORM = ["Name_norm", "Age", "Gender_norm", "Blood_norm"]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons stricts (lignes identiques)."""
    return df.drop_duplicates()

def group_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe les lignes par patient (après normalisation des champs clés).
    Chaque groupe devient un patient avec plusieurs admissions possibles.
    """
    df = df.copy()
    df["Name_norm"]   = df["Name"].apply(normalize_name)
    df["Gender_norm"] = df["Gender"].apply(normalize_gender)
    df["Blood_norm"]  = df["Blood Type"].apply(normalize_blood)

    # On garde toutes les colonnes sauf celles utilisées pour le groupement
    admission_cols = [c for c in df.columns if c not in ["Name","Age","Gender","Blood Type"] + GROUP_COLS_NORM]

    grouped = (
        df.groupby(GROUP_COLS_NORM, dropna=False)[admission_cols]
          .apply(lambda x: x.to_dict(orient="records"))  # liste de dicts = admissions
          .reset_index(name="Admissions")
    )

    # On remet des noms propres pour l’export final
    grouped["Name"] = grouped["Name_norm"]
    grouped["Gender"] = grouped["Gender_norm"]
    grouped["Blood Type"] = grouped["Blood_norm"]

    return grouped[["Name","Age","Gender","Blood Type","Admissions"]]


def make_docs(grouped: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convertit les groupes pandas en documents JSON.
    - Déduplication stricte des admissions identiques
    - Tri des admissions par Date of Admission
    """
    docs: List[Dict[str, Any]] = []
    for _, row in grouped.iterrows():
        admissions = list(row["Admissions"])

        # On crée une "signature" pour repérer les doublons exacts d’admission
        seen = set()
        unique_adm = []
        for a in admissions:
            sig = tuple(_clean_str(str(v)) for v in a.values())
            if sig in seen:
                continue
            seen.add(sig)
            unique_adm.append(a)

        # Tri chronologique des admissions
        def _adm_key(a):
            return str(a.get("Date of Admission","")) or ""

        unique_adm.sort(key=_adm_key)

        doc = {
            "Name": row["Name"],
            "Age": int(row["Age"]) if pd.notna(row["Age"]) else None,
            "Gender": row["Gender"],
            "Blood Type": row["Blood Type"],
            "Admissions": unique_adm
        }
        docs.append(doc)
    return docs


# =======================================================================
# 🧪 DEBUG : détecter les patients à plusieurs admissions
# =======================================================================

def debug_multi_admissions_from_csv(df: pd.DataFrame, reports_dir: Path) -> None:
    """
    Vérifie s’il existe des patients qui apparaissent plusieurs fois dans le CSV
    (donc plusieurs admissions).
    """
    tmp = df.copy()
    tmp["Name_norm"]   = tmp["Name"].apply(normalize_name)
    tmp["Gender_norm"] = tmp["Gender"].apply(normalize_gender)
    tmp["Blood_norm"]  = tmp["Blood Type"].apply(normalize_blood)

    # Compte combien de fois chaque patient normalisé apparaît
    grp_cols = ["Name_norm", "Age", "Gender_norm", "Blood_norm"]
    counts = tmp.groupby(grp_cols, dropna=False).size().reset_index(name="rows")

    multi = counts[counts["rows"] > 1]
    print(f"🔬 DEBUG CSV — groupes>1: {len(multi)} patients multi-admissions")
    if len(multi) > 0:
        sample_path = reports_dir / "multi_admissions_sample.csv"
        tmp.merge(multi, on=grp_cols, how="inner").to_csv(sample_path, index=False)
        print(f"🧪 Exemple écrit: {sample_path.resolve()}")


# =======================================================================
# 💾 EXPORT ET INSERTION DANS MONGO
# =======================================================================

def export_patients_json(docs: List[Dict[str, Any]], reports_dir: Path) -> Path:
    """Sauvegarde tous les patients au format JSON."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "patients.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"📝 Export écrit: {out.resolve()}")
    return out


def upsert_mongo(docs: List[Dict[str, Any]], client: MongoClient, db_name: str, coll_name: str = "patients") -> None:
    """
    Met à jour ou insère chaque patient dans MongoDB.
    La clé unique est basée sur (Name, Age, Gender, Blood Type).
    """
    col = client[db_name][coll_name]
    ops = []

    for d in docs:
        filt = {
            "Name": d.get("Name"),
            "Age": d.get("Age"),
            "Gender": d.get("Gender"),
            "Blood Type": d.get("Blood Type"),
        }
        ops.append(UpdateOne(filt, {"$set": d}, upsert=True))

    if not ops:
        print("⚠️ Aucun document à insérer.")
        return

    try:
        res = col.bulk_write(ops, ordered=False)
        upserts = len(getattr(res, "upserted_ids", {}) or {})
        print(f"✅ Upserts: {upserts}, Modifiés: {res.modified_count}, Mis à jour (matched): {res.matched_count}")
    except BulkWriteError as e:
        print("❌ Erreur BulkWrite:", json.dumps(e.details, indent=2))


# =======================================================================
# 🚀 MAIN : enchaînement complet
# =======================================================================

def main():
    # 1️⃣ Charger la config
    CSV_PATH, REPORTS_DIR, MONGO_URI, DB_NAME, ENABLE_MONGO_INSERT = load_config()

    # 2️⃣ Lire le CSV et résumer
    df = read_data(CSV_PATH)
    print("✅ Lecture terminée")
    summarize(df)

    # 3️⃣ Supprimer les doublons stricts
    df = clean(df)
    print("✅ Nettoyage terminé")

    # 4️⃣ Vérifier s’il y a des multi-admissions
    debug_multi_admissions_from_csv(df, REPORTS_DIR)

    # 5️⃣ Regrouper et construire les docs
    grouped = group_patients(df)
    print(f"✅ Groupement terminé -> {len(grouped)} patients uniques")
    docs = make_docs(grouped)
    print(f"📦 Documents prêts: {len(docs)}")

    # 6️⃣ Export JSON
    export_patients_json(docs, REPORTS_DIR)

    # 7️⃣ Envoi vers MongoDB (optionnel)
    if ENABLE_MONGO_INSERT:
        client = get_client(MONGO_URI)
        wait_for_mongo(client)
        upsert_mongo(docs, client, DB_NAME, "patients")
    else:
        print("ℹ️ Insertion Mongo désactivée.")

# Point d’entrée du script
if __name__ == "__main__":
    main()
