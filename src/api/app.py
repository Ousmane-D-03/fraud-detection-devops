"""
app.py - API Flask pour la détection de fraude

Cette API expose le modèle ML via des endpoints REST :
- GET /health : Vérifier que l'API fonctionne
- POST /predict : Prédire si une transaction est frauduleuse
- GET /info : Informations sur le modèle

"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# ====================
# CONFIGURATION
# ====================

# Créer l'application Flask
app = Flask(__name__)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins vers les modèles
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detector.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.pkl"

# ====================
# CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ====================

logger.info("Chargement des modèles...")

try:
    # Charger le modèle
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Modèle chargé : {type(model).__name__}")
    
    # Charger le scaler
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"✅ Scaler chargé : {type(scaler).__name__}")
    
    # Charger les métadonnées (optionnel)
    try:
        metadata = joblib.load(METADATA_PATH)
        logger.info(f"✅ Métadonnées chargées")
    except Exception as e:
        metadata = None
        logger.warning(f"⚠️ Métadonnées non chargées : {e}")
    
    logger.info("🚀 Tous les modèles chargés avec succès !")
    
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles : {e}")
    raise


# ====================
# ENDPOINT 1 : HEALTH CHECK
# ====================

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint pour vérifier que l'API fonctionne
    
    Usage:
        GET http://localhost:5000/health
    
    Response:
        {
            "status": "healthy",
            "timestamp": "2025-10-21T10:30:00",
            "model_loaded": true
        }
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }), 200


# ====================
# ENDPOINT 2 : INFORMATIONS MODÈLE
# ====================

@app.route('/info', methods=['GET'])
def info():
    """
    Endpoint pour obtenir des informations sur le modèle
    
    Usage:
        GET http://localhost:5000/info
    
    Response:
        {
            "model_type": "RandomForestClassifier",
            "n_features": 30,
            "performance": {...}
        }
    """
    info_data = {
        "model_type": type(model).__name__,
        "n_features_expected": 30,
        "scaler_type": type(scaler).__name__
    }
    
    # Ajouter métadonnées si disponibles
    if metadata:
        info_data["performance"] = {
            "f1_score": metadata.get('f1_score'),
            "recall": metadata.get('recall'),
            "precision": metadata.get('precision'),
            "training_date": metadata.get('training_date')
        }
        info_data["model_name"] = metadata.get('model_name')
    
    return jsonify(info_data), 200


# ====================
# ENDPOINT 3 : PRÉDICTION
# ====================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour prédire si une transaction est frauduleuse
    
    Usage:
        POST http://localhost:5000/predict
        Content-Type: application/json
        
        Body:
        {
            "transaction": [0, -1.359, -0.072, ..., 149.62]
        }
    
    Response succès (200):
        {
            "prediction": 0,
            "prediction_label": "Normal",
            "fraud_probability": 0.05,
            "timestamp": "2025-10-21T10:30:00"
        }
    
    Response erreur (400):
        {
            "error": "Message d'erreur",
            "timestamp": "2025-10-21T10:30:00"
        }
    """
    try:
        # 1. Récupérer les données JSON
        data = request.get_json()
        
        # 2. Validation : JSON existe ?
        if not data:
            logger.warning("Requête sans données JSON")
            return jsonify({
                "error": "Pas de données JSON fournies",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # 3. Validation : Clé "transaction" existe ?
        if "transaction" not in data:
            logger.warning("Clé 'transaction' manquante")
            return jsonify({
                "error": "Clé 'transaction' manquante dans le JSON",
                "example": {
                    "transaction": [0, -1.359, -0.072, "...", 149.62]
                },
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # 4. Extraire la transaction
        transaction = data["transaction"]
        
        # 5. Validation : C'est une liste ?
        if not isinstance(transaction, list):
            logger.warning(f"Transaction n'est pas une liste : {type(transaction)}")
            return jsonify({
                "error": f"Transaction doit être une liste, pas {type(transaction).__name__}",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # 6. Validation : 30 features ?
        if len(transaction) != 30:
            logger.warning(f"Nombre de features incorrect : {len(transaction)}")
            return jsonify({
                "error": f"Transaction doit contenir 30 features, reçu {len(transaction)}",
                "expected_format": "Time, V1, V2, ..., V28, Amount (30 valeurs)",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # 7. Convertir en array NumPy
        try:
            transaction_array = np.array(transaction).reshape(1, -1)
        except Exception as e:
            logger.error(f"Erreur conversion NumPy : {e}")
            return jsonify({
                "error": f"Erreur lors de la conversion en array : {str(e)}",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # 8. Normaliser avec le scaler
        try:
            transaction_scaled = scaler.transform(transaction_array)
        except Exception as e:
            logger.error(f"Erreur normalisation : {e}")
            return jsonify({
                "error": f"Erreur lors de la normalisation : {str(e)}",
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # 9. Prédiction
        try:
            prediction = int(model.predict(transaction_scaled)[0])
            probabilities = model.predict_proba(transaction_scaled)[0]
            fraud_probability = float(probabilities[1])
        except Exception as e:
            logger.error(f"Erreur prédiction : {e}")
            return jsonify({
                "error": f"Erreur lors de la prédiction : {str(e)}",
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # 10. Préparer la réponse
        response = {
            "prediction": prediction,
            "prediction_label": "Fraude" if prediction == 1 else "Normal",
            "fraud_probability": round(fraud_probability, 4),
            "normal_probability": round(probabilities[0], 4),
            "timestamp": datetime.now().isoformat()
        }
        
        # Ajouter alerte si probabilité élevée
        if fraud_probability > 0.7:
            response["alert"] = "⚠️ Probabilité de fraude élevée !"
        
        # Logger la prédiction
        logger.info(
            f"Prédiction : {prediction} "
            f"(Proba fraude: {fraud_probability:.4f})"
        )
        
        return jsonify(response), 200
    
    except Exception as e:
        # Erreur générique non gérée
        logger.error(f"Erreur inattendue : {e}", exc_info=True)
        return jsonify({
            "error": "Erreur interne du serveur",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# ====================
# ENDPOINT 4 : BATCH PREDICTION (bonus)
# ====================

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Endpoint pour prédire plusieurs transactions à la fois
    
    Usage:
        POST http://localhost:5000/predict/batch
        
        Body:
        {
            "transactions": [
                [0, -1.359, ..., 149.62],
                [406, 1.191, ..., 2.69]
            ]
        }
    
    Response:
        {
            "predictions": [0, 0],
            "fraud_probabilities": [0.05, 0.12],
            "count": 2
        }
    """
    try:
        data = request.get_json()
        
        if not data or "transactions" not in data:
            return jsonify({
                "error": "Clé 'transactions' manquante",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        transactions = data["transactions"]
        
        # Validation
        if not isinstance(transactions, list):
            return jsonify({
                "error": "transactions doit être une liste",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        if len(transactions) == 0:
            return jsonify({
                "error": "Liste de transactions vide",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Vérifier chaque transaction
        for i, trans in enumerate(transactions):
            if len(trans) != 30:
                return jsonify({
                    "error": f"Transaction {i} : {len(trans)} features au lieu de 30",
                    "timestamp": datetime.now().isoformat()
                }), 400
        
        # Convertir et normaliser
        transactions_array = np.array(transactions)
        transactions_scaled = scaler.transform(transactions_array)
        
        # Prédictions
        predictions = model.predict(transactions_scaled).tolist()
        probabilities = model.predict_proba(transactions_scaled)
        fraud_probs = [round(float(p[1]), 4) for p in probabilities]
        
        response = {
            "predictions": predictions,
            "fraud_probabilities": fraud_probs,
            "count": len(predictions),
            "fraud_count": sum(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Batch prédiction : {len(predictions)} transactions")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Erreur batch prediction : {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# ====================
# GESTION D'ERREURS
# ====================

@app.errorhandler(404)
def not_found(error):
    """Gestion des routes non trouvées"""
    return jsonify({
        "error": "Endpoint non trouvé",
        "available_endpoints": [
            "GET /health",
            "GET /info",
            "POST /predict",
            "POST /predict/batch"
        ],
        "timestamp": datetime.now().isoformat()
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Gestion des méthodes HTTP non autorisées"""
    return jsonify({
        "error": "Méthode HTTP non autorisée",
        "timestamp": datetime.now().isoformat()
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs internes"""
    logger.error(f"Erreur 500 : {error}")
    return jsonify({
        "error": "Erreur interne du serveur",
        "timestamp": datetime.now().isoformat()
    }), 500


# ====================
# POINT D'ENTRÉE
# ====================

if __name__ == '__main__':
    """
    Lancer l'API en mode développement
    
    Usage:
        python src/api/app.py
    
    L'API sera accessible sur http://localhost:5000
    """
    logger.info("="*50)
    logger.info("🚀 DÉMARRAGE API DÉTECTION DE FRAUDE")
    logger.info("="*50)
    logger.info(f"Modèle : {type(model).__name__}")
    logger.info(f"Features attendues : 30")
    if metadata:
        logger.info(f"F1-Score : {metadata.get('f1_score', 'N/A')}")
    logger.info("="*50)
    logger.info("Endpoints disponibles :")
    logger.info("  - GET  /health")
    logger.info("  - GET  /info")
    logger.info("  - POST /predict")
    logger.info("  - POST /predict/batch")
    logger.info("="*50)
    
    # Lancer le serveur
    # debug=True : Recharge automatiquement si le code change
    # host='0.0.0.0' : Accessible depuis autres machines
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
