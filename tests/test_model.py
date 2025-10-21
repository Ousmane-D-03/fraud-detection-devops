"""
test_model.py - Tests unitaires pour le modèle de détection de fraude

Ce fichier teste que :
1. Le modèle se charge correctement
2. Le scaler se charge correctement
3. Le modèle accepte le bon format d'input
4. Le modèle retourne le bon format d'output
5. Les prédictions sont cohérentes (reproductibles)

Auteur : Ton Nom
Date : Jour 4
"""

import pytest
import joblib
import numpy as np
import os
from pathlib import Path

# ====================
# CONFIGURATION
# ====================

# Chemin vers le dossier models (depuis tests/)
MODEL_PATH = Path(__file__).parent.parent / "models" / "fraud_detector.pkl"
SCALER_PATH = Path(__file__).parent.parent / "models" / "scaler.pkl"
METADATA_PATH = Path(__file__).parent.parent / "models" / "metadata.pkl"


# ====================
# FIXTURES (données partagées entre tests)
# ====================

@pytest.fixture
def model():
    """
    Charge le modèle pour les tests
    
    Fixture = fonction qui s'exécute AVANT chaque test
    et fournit des données communes
    """
    if not MODEL_PATH.exists():
        pytest.skip("Modèle non trouvé. Exécuter train.py d'abord.")
    
    return joblib.load(MODEL_PATH)


@pytest.fixture
def scaler():
    """
    Charge le scaler pour les tests
    """
    if not SCALER_PATH.exists():
        pytest.skip("Scaler non trouvé. Exécuter train.py d'abord.")
    
    return joblib.load(SCALER_PATH)


@pytest.fixture
def sample_transaction():
    """
    Crée une transaction exemple pour les tests
    
    Format : [Time, V1, V2, ..., V28, Amount]
    30 features au total
    """
    # Transaction normale typique
    transaction = np.array([
        0,          # Time
        -1.359807,  # V1
        -0.072781,  # V2
        2.536347,   # V3
        1.378155,   # V4
        -0.338321,  # V5
        0.462388,   # V6
        0.239599,   # V7
        0.098698,   # V8
        0.363787,   # V9
        0.090794,   # V10
        -0.551600,  # V11
        -0.617801,  # V12
        -0.991390,  # V13
        -0.311169,  # V14
        1.468177,   # V15
        -0.470401,  # V16
        0.207971,   # V17
        0.025791,   # V18
        0.403993,   # V19
        0.251412,   # V20
        -0.018307,  # V21
        0.277838,   # V22
        -0.110474,  # V23
        0.066928,   # V24
        0.128539,   # V25
        -0.189115,  # V26
        0.133558,   # V27
        -0.021053,  # V28
        149.62      # Amount
    ]).reshape(1, -1)  # Reshape pour format 2D
    
    return transaction


@pytest.fixture
def sample_fraud_transaction():
    """
    Transaction frauduleuse typique
    """
    transaction = np.array([
        406,        # Time
        2.311543,   # V1 (différent d'une normale)
        0.876857,   # V2
        1.548718,   # V3
        0.403034,   # V4
        -0.407193,  # V5
        0.095921,   # V6
        0.592941,   # V7
        -0.270533,  # V8
        0.817739,   # V9
        0.753074,   # V10
        -0.822843,  # V11
        0.538196,   # V12
        1.345852,   # V13
        -1.119670,  # V14 (pattern fraude)
        0.175121,   # V15
        -0.451449,  # V16
        -0.237033,  # V17
        -0.038195,  # V18
        0.803487,   # V19
        0.408542,   # V20
        -0.009431,  # V21
        0.798278,   # V22
        -0.137458,  # V23
        0.141267,   # V24
        -0.206010,  # V25
        0.502292,   # V26
        0.219422,   # V27
        0.215153,   # V28
        1.00        # Amount (petit montant = test fraude)
    ]).reshape(1, -1)
    
    return transaction


# ====================
# TEST 1 : CHARGEMENT MODÈLE
# ====================

def test_model_file_exists():
    """
    Vérifie que le fichier modèle existe
    
    Test le plus basique : le fichier .pkl est là ?
    """
    assert MODEL_PATH.exists(), (
        f"Fichier modèle introuvable : {MODEL_PATH}\n"
        "Exécuter 'python src/ml/train.py' d'abord"
    )
    
    # Vérifier taille (doit être > 1KB)
    file_size = MODEL_PATH.stat().st_size
    assert file_size > 1000, (
        f"Fichier modèle trop petit ({file_size} bytes). "
        "Probablement corrompu."
    )


def test_model_loads_successfully(model):
    """
    Vérifie que le modèle se charge sans erreur
    
    Test : joblib.load() fonctionne ?
    """
    assert model is not None, "Modèle chargé est None"
    
    # Vérifier que c'est bien un modèle sklearn
    assert hasattr(model, 'predict'), (
        "L'objet chargé n'a pas de méthode predict(). "
        "Ce n'est pas un modèle sklearn valide."
    )
    assert hasattr(model, 'predict_proba'), (
        "L'objet chargé n'a pas de méthode predict_proba()"
    )


def test_model_type(model):
    """
    Vérifie le type du modèle
    
    Test : C'est bien un RandomForest ou LogisticRegression ?
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    assert isinstance(model, (RandomForestClassifier, LogisticRegression)), (
        f"Type de modèle inattendu : {type(model)}"
    )


# ====================
# TEST 2 : CHARGEMENT SCALER
# ====================

def test_scaler_file_exists():
    """
    Vérifie que le fichier scaler existe
    """
    assert SCALER_PATH.exists(), (
        f"Fichier scaler introuvable : {SCALER_PATH}\n"
        "Exécuter 'python src/ml/train.py' d'abord"
    )


def test_scaler_loads_successfully(scaler):
    """
    Vérifie que le scaler se charge sans erreur
    """
    assert scaler is not None, "Scaler chargé est None"
    assert hasattr(scaler, 'transform'), (
        "L'objet chargé n'a pas de méthode transform()"
    )


def test_scaler_has_correct_features(scaler):
    """
    Vérifie que le scaler a été entraîné sur 30 features
    
    Test : Le scaler attend bien 30 colonnes ?
    """
    assert scaler.n_features_in_ == 30, (
        f"Scaler entraîné sur {scaler.n_features_in_} features, "
        "attendu 30 (Time + V1-V28 + Amount)"
    )


# ====================
# TEST 3 : FORMAT INPUT
# ====================

def test_model_accepts_correct_shape(model, scaler, sample_transaction):
    """
    Vérifie que le modèle accepte le bon format d'input
    
    Test : Shape (1, 30) fonctionne ?
    """
    # Scaler la transaction
    scaled = scaler.transform(sample_transaction)
    
    # Prédire (ne doit pas lever d'exception)
    try:
        prediction = model.predict(scaled)
        assert True  # Si on arrive ici, c'est bon
    except Exception as e:
        pytest.fail(f"Erreur lors de la prédiction : {e}")


def test_model_rejects_wrong_shape(model, scaler):
    """
    Vérifie que le modèle rejette un mauvais format
    
    Test : Shape incorrecte (ex: 28 features) → erreur ?
    """
    # Transaction avec seulement 28 features (manque 2)
    wrong_transaction = np.random.randn(1, 28)
    
    with pytest.raises(ValueError):
        scaler.transform(wrong_transaction)


def test_model_rejects_1d_array(model, scaler):
    """
    Vérifie que le modèle rejette un array 1D
    
    Test : Shape (30,) au lieu de (1, 30) → erreur ?
    """
    # Array 1D (mauvais format)
    wrong_transaction = np.random.randn(30)
    
    with pytest.raises(ValueError):
        scaler.transform(wrong_transaction)


# ====================
# TEST 4 : FORMAT OUTPUT
# ====================

def test_model_output_is_binary(model, scaler, sample_transaction):
    """
    Vérifie que le modèle retourne 0 ou 1
    
    Test : Prédiction ∈ {0, 1} ?
    """
    scaled = scaler.transform(sample_transaction)
    prediction = model.predict(scaled)[0]
    
    assert prediction in [0, 1], (
        f"Prédiction invalide : {prediction}. "
        "Attendu 0 (Normal) ou 1 (Fraude)"
    )


def test_model_output_shape(model, scaler, sample_transaction):
    """
    Vérifie que le modèle retourne un array de shape (1,)
    
    Test : Une seule prédiction pour une transaction ?
    """
    scaled = scaler.transform(sample_transaction)
    prediction = model.predict(scaled)
    
    assert prediction.shape == (1,), (
        f"Shape de prédiction incorrect : {prediction.shape}. "
        "Attendu (1,)"
    )


def test_model_proba_output(model, scaler, sample_transaction):
    """
    Vérifie que predict_proba retourne des probabilités valides
    
    Test : Probabilités ∈ [0, 1] et somme = 1 ?
    """
    scaled = scaler.transform(sample_transaction)
    probas = model.predict_proba(scaled)[0]
    
    # Doit avoir 2 probabilités (classe 0 et classe 1)
    assert len(probas) == 2, (
        f"Nombre de probabilités incorrect : {len(probas)}. Attendu 2"
    )
    
    # Chaque probabilité entre 0 et 1
    assert all(0 <= p <= 1 for p in probas), (
        f"Probabilités hors intervalle [0, 1] : {probas}"
    )
    
    # Somme = 1 (±0.001 tolérance numérique)
    assert abs(sum(probas) - 1.0) < 0.001, (
        f"Somme des probabilités ≠ 1 : {sum(probas)}"
    )


# ====================
# TEST 5 : COHÉRENCE PRÉDICTIONS
# ====================

def test_model_predictions_are_deterministic(model, scaler, sample_transaction):
    """
    Vérifie que les prédictions sont reproductibles
    
    Test : Même input → même output ?
    """
    scaled = scaler.transform(sample_transaction)
    
    # Prédire 3 fois
    pred1 = model.predict(scaled)[0]
    pred2 = model.predict(scaled)[0]
    pred3 = model.predict(scaled)[0]
    
    assert pred1 == pred2 == pred3, (
        f"Prédictions non déterministes : {pred1}, {pred2}, {pred3}"
    )


def test_model_detects_normal_transaction(model, scaler, sample_transaction):
    """
    Vérifie que le modèle détecte une transaction normale
    
    Test : Transaction typique normale → prédiction 0 ?
    """
    scaled = scaler.transform(sample_transaction)
    prediction = model.predict(scaled)[0]
    proba_fraud = model.predict_proba(scaled)[0][1]
    
    # On s'attend à prédiction = 0 (Normal)
    # Mais on tolère une erreur si probabilité < 0.5
    assert prediction == 0 or proba_fraud < 0.5, (
        f"Transaction normale mal classée. "
        f"Prédiction: {prediction}, Proba fraude: {proba_fraud:.3f}"
    )


def test_model_detects_fraud_transaction(model, scaler, sample_fraud_transaction):
    """
    Vérifie que le modèle détecte une transaction frauduleuse
    
    Test : Transaction typique fraude → prédiction 1 ?
    """
    scaled = scaler.transform(sample_fraud_transaction)
    prediction = model.predict(scaled)[0]
    proba_fraud = model.predict_proba(scaled)[0][1]
    
    # On s'attend à prédiction = 1 (Fraude)
    # Ou au moins probabilité > 0.5
    assert prediction == 1 or proba_fraud > 0.5, (
        f"Transaction fraude mal classée. "
        f"Prédiction: {prediction}, Proba fraude: {proba_fraud:.3f}"
    )


# ====================
# TEST 6 : MÉTADONNÉES
# ====================

def test_metadata_exists():
    """
    Vérifie que le fichier metadata existe
    """
    assert METADATA_PATH.exists(), (
        f"Fichier metadata introuvable : {METADATA_PATH}"
    )


def test_metadata_content():
    """
    Vérifie le contenu des métadonnées
    """
    if not METADATA_PATH.exists():
        pytest.skip("Metadata non trouvé")
    
    metadata = joblib.load(METADATA_PATH)
    
    # Vérifier clés obligatoires
    required_keys = ['model_name', 'f1_score', 'training_date']
    for key in required_keys:
        assert key in metadata, f"Clé manquante dans metadata : {key}"
    
    # Vérifier F1-score raisonnable
    f1 = metadata['f1_score']
    assert 0 <= f1 <= 1, f"F1-score invalide : {f1}"
    assert f1 > 0.5, (
        f"F1-score trop bas : {f1}. "
        "Le modèle semble mal entraîné."
    )


# ====================
# TEST 7 : PERFORMANCE MINIMUM
# ====================

def test_model_minimum_performance():
    """
    Vérifie que le modèle a des performances minimales acceptables
    
    Test : F1-score > 0.70 ?
    """
    if not METADATA_PATH.exists():
        pytest.skip("Metadata non trouvé")
    
    metadata = joblib.load(METADATA_PATH)
    f1_score = metadata['f1_score']
    
    MIN_F1 = 0.70  # Seuil minimum acceptable
    
    assert f1_score >= MIN_F1, (
        f"Performance insuffisante ! "
        f"F1-score: {f1_score:.4f} < {MIN_F1}. "
        "Ré-entraîner le modèle avec meilleurs hyperparamètres."
    )


# ====================
# TEST 8 : BATCH PREDICTIONS
# ====================

def test_model_batch_predictions(model, scaler):
    """
    Vérifie que le modèle gère plusieurs transactions à la fois
    
    Test : Batch de 10 transactions → 10 prédictions ?
    """
    # Créer 10 transactions aléatoires
    batch = np.random.randn(10, 30)
    scaled_batch = scaler.transform(batch)
    
    predictions = model.predict(scaled_batch)
    
    assert predictions.shape == (10,), (
        f"Shape de batch incorrect : {predictions.shape}. Attendu (10,)"
    )
    
    # Toutes les prédictions sont 0 ou 1
    assert all(p in [0, 1] for p in predictions), (
        "Certaines prédictions ne sont pas binaires"
    )


# ====================
# INFORMATIONS
# ====================

def test_print_model_info(model, capsys):
    """
    Affiche des informations sur le modèle (pas vraiment un test)
    
    Utile pour debug
    """
    print("\n" + "="*50)
    print("INFORMATIONS MODÈLE")
    print("="*50)
    print(f"Type : {type(model).__name__}")
    
    if hasattr(model, 'n_estimators'):
        print(f"Nombre d'arbres : {model.n_estimators}")
    
    if hasattr(model, 'max_depth'):
        print(f"Profondeur max : {model.max_depth}")
    
    if METADATA_PATH.exists():
        metadata = joblib.load(METADATA_PATH)
        print(f"\nPerformances :")
        print(f"  - F1-Score : {metadata['f1_score']:.4f}")
        print(f"  - Recall : {metadata.get('recall', 'N/A')}")
        print(f"  - Precision : {metadata.get('precision', 'N/A')}")
        print(f"  - Date entraînement : {metadata['training_date']}")
    
    print("="*50)
    
    # Ce test passe toujours (juste informatif)
    assert True
