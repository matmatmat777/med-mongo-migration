# ---------- Image de l'app d'ingestion ----------
# On fige Python à 3.11 pour être compatible avec pandas sur toutes les plateformes.
# Ainsi, on peut rester en Python 3.13 sur Windows sans souci : tout se passe ici.

# Note : le "syntax=" permet d'utiliser des features modernes du Dockerfile si besoin
# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Répertoire de travail dans l'image
WORKDIR /app

# (Optionnel) Outils de build pour accélérer/assurer la compilation de certaines libs
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# On copie le requirements en premier pour tirer parti du cache Docker
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# on copie le code depuis la racine
COPY src/ingest/app.py ./app.py
COPY src/ingest/config.yaml ./config.yaml

# Le point d'entrée réel est défini dans docker-compose.yml via "command"
CMD ["python", "app.py"]
