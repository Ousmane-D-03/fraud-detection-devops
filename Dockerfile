# ==========================================
# STAGE 1 : BASE IMAGE
# ==========================================
# Utiliser Python 3.11 slim (version légère)
FROM python:3.11-slim as base

# Métadonnées de l'image
LABEL maintainer="ousmanesarrd@gmail.com"
LABEL description="API de détection de fraude avec ML"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Créer utilisateur non-root (sécurité)
RUN useradd -m -u 1000 apiuser && \
    mkdir -p /app /app/models /app/logs && \
    chown -R apiuser:apiuser /app

# Définir le répertoire de travail
WORKDIR /app

# ==========================================
# STAGE 2 : DEPENDENCIES
# ==========================================
FROM base as dependencies

# Installer dépendances système si nécessaires
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt
COPY --chown=apiuser:apiuser requirements.txt .

# Installer dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 3 : APPLICATION
# ==========================================
FROM base as application

# Copier dépendances depuis stage précédent
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copier le code source
COPY --chown=apiuser:apiuser src/ ./src/

# Copier les modèles
COPY --chown=apiuser:apiuser models/ ./models/

# Vérifier que les modèles existent
RUN if [ ! -f ./models/fraud_detector.pkl ]; then \
        echo "ERREUR: fraud_detector.pkl manquant !"; \
        exit 1; \
    fi && \
    if [ ! -f ./models/scaler.pkl ]; then \
        echo "ERREUR: scaler.pkl manquant !"; \
        exit 1; \
    fi

# Changer vers utilisateur non-root
USER apiuser

# Port exposé
EXPOSE 5000

# Healthcheck (Docker vérifie que l'API marche)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Point d'entrée
CMD ["python", "src/api/app.py"]