"""
train.py - Entraînement du modèle de détection de fraude

Ce fichier fait 5 choses simples :
1. Charger les données (le CSV)
2. Préparer les données (séparer X et y, normaliser)
3. Entraîner 2 modèles (Logistic Regression et Random Forest)
4. Comparer les performances
5. Sauvegarder le meilleur modèle

Auteur : Ton Nom
Date : Jour 3-4
"""

# ====================
# IMPORTS (bibliothèques nécessaires)
# ====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("DÉTECTION DE FRAUDE - ENTRAÎNEMENT DU MODÈLE")
print("=" * 50)

# ====================
# ÉTAPE 1 : CHARGER LES DONNÉES
# ====================
print("\n[1/5] Chargement du dataset...")

# Charger le fichier CSV
df = pd.read_csv('/home/ousmane/projects/fraud-detection-devops/data/creditcard.csv')

print(f"✅ Dataset chargé : {df.shape[0]} transactions, {df.shape[1]} colonnes")
print(f"   - Transactions normales : {(df['Class']==0).sum()}")
print(f"   - Transactions frauduleuses : {(df['Class']==1).sum()}")

# ====================
# ÉTAPE 2 : PRÉPARER LES DONNÉES
# ====================
print("\n[2/5] Préparation des données...")

# 2.1 Séparer X (features) et y (target)
# X = toutes les colonnes SAUF 'Class'
# y = seulement la colonne 'Class'
X = df.drop('Class', axis=1)  # axis=1 = supprimer colonne
y = df['Class']

print(f"✅ Features (X) : {X.shape}")
print(f"✅ Target (y) : {y.shape}")

# 2.2 Diviser en train/test (80% train, 20% test)
# stratify=y : garde le même ratio fraude/normal dans train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour test
    random_state=42,    # Pour avoir toujours le même split
    stratify=y          # Garde ratio 0.17% dans train et test
)

print(f"✅ Train set : {X_train.shape[0]} transactions")
print(f"✅ Test set : {X_test.shape[0]} transactions")

# 2.3 Normaliser les données (StandardScaler)
# Formule : (x - moyenne) / écart-type
# Résultat : tous les nombres entre -3 et +3
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Données normalisées (moyenne=0, écart-type=1)")

# Sauvegarder le scaler (important pour plus tard !)
joblib.dump(scaler, '/home/ousmane/projects/fraud-detection-devops/models/scaler.pkl')
print(f"✅ Scaler sauvegardé dans models/scaler.pkl")

# ====================
# ÉTAPE 3 : ENTRAÎNER LES MODÈLES
# ====================
print("\n[3/5] Entraînement des modèles...")

# ------------------------------
# MODÈLE 1 : LOGISTIC REGRESSION
# ------------------------------
print("\n📊 Modèle 1 : Logistic Regression")
print("-" * 40)

# Créer le modèle
# class_weight='balanced' : donne plus d'importance aux fraudes (rare)
lr_model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000  # Nombre d'itérations pour apprendre
)

# Entraîner (le modèle apprend ici !)
print("⏳ Entraînement en cours...")
lr_model.fit(X_train_scaled, y_train)
print("✅ Entraînement terminé !")

# Prédire sur le test
y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]  # Probabilité classe 1

# Calculer les performances
lr_f1 = f1_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_proba_lr)

print(f"\n📈 Performances Logistic Regression :")
print(f"   - F1-Score    : {lr_f1:.4f}")
print(f"   - Recall      : {lr_recall:.4f} ({lr_recall*100:.1f}% fraudes détectées)")
print(f"   - Precision   : {lr_precision:.4f} ({lr_precision*100:.1f}% alertes correctes)")
print(f"   - AUC-ROC     : {lr_auc:.4f}")

# ------------------------------
# MODÈLE 2 : RANDOM FOREST
# ------------------------------
print("\n🌲 Modèle 2 : Random Forest")
print("-" * 40)

# Créer le modèle
# n_estimators=100 : 100 arbres de décision
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # Utilise tous les CPU
)

# Entraîner
print("⏳ Entraînement en cours (plus long que Logistic Regression)...")
rf_model.fit(X_train_scaled, y_train)
print("✅ Entraînement terminé !")

# Prédire
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Performances
rf_f1 = f1_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_proba_rf)

print(f"\n📈 Performances Random Forest :")
print(f"   - F1-Score    : {rf_f1:.4f}")
print(f"   - Recall      : {rf_recall:.4f} ({rf_recall*100:.1f}% fraudes détectées)")
print(f"   - Precision   : {rf_precision:.4f} ({rf_precision*100:.1f}% alertes correctes)")
print(f"   - AUC-ROC     : {rf_auc:.4f}")

# ====================
# ÉTAPE 4 : COMPARER LES MODÈLES
# ====================
print("\n[4/5] Comparaison des modèles...")
print("=" * 50)

# Tableau comparatif
comparison = pd.DataFrame({
    'Modèle': ['Logistic Regression', 'Random Forest'],
    'F1-Score': [lr_f1, rf_f1],
    'Recall': [lr_recall, rf_recall],
    'Precision': [lr_precision, rf_precision],
    'AUC-ROC': [lr_auc, rf_auc]
})

print("\n📊 COMPARAISON DES PERFORMANCES :")
print(comparison.to_string(index=False))

# Choisir le meilleur modèle (basé sur F1-Score)
if rf_f1 > lr_f1:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_f1 = rf_f1
    y_pred_best = y_pred_rf
else:
    best_model = lr_model
    best_model_name = "Logistic Regression"
    best_f1 = lr_f1
    y_pred_best = y_pred_lr

print(f"\n🏆 MEILLEUR MODÈLE : {best_model_name} (F1={best_f1:.4f})")

# ====================
# ÉTAPE 5 : SAUVEGARDER LE MEILLEUR MODÈLE
# ====================
print("\n[5/5] Sauvegarde du modèle...")

# Sauvegarder
model_path = '/home/ousmane/projects/fraud-detection-devops/models/fraud_detector.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Modèle sauvegardé dans : {model_path}")

# Sauvegarder aussi les métadonnées
metadata = {
    'model_name': best_model_name,
    'f1_score': best_f1,
    'recall': recall_score(y_test, y_pred_best),
    'precision': precision_score(y_test, y_pred_best),
    'auc_roc': roc_auc_score(y_test, y_proba_rf if best_model_name == "Random Forest" else y_proba_lr),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_size': len(X_train),
    'test_size': len(X_test)
}

joblib.dump(metadata, '/home/ousmane/projects/fraud-detection-devops/models/metadata.pkl')
print(f"✅ Métadonnées sauvegardées dans : models/metadata.pkl")

# ====================
# VISUALISATIONS
# ====================
print("\n📊 Génération des visualisations...")

# Créer dossier pour les images si n'existe pas
import os
os.makedirs('./home/ousmane/projects/fraud-detection-devops/docs/images', exist_ok=True)

# 1. Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de Confusion - {best_model_name}')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.savefig('/home/ousmane/projects/fraud-detection-devops/docs/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Matrice de confusion sauvegardée")

# 2. Feature Importance (si Random Forest)
if best_model_name == "Random Forest":
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Top 15 Features les plus importantes')
    plt.xlabel('Importance')
    plt.savefig('/home/ousmane/projects/fraud-detection-devops/docs/images/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Feature importance sauvegardée")

plt.close('all')

# ====================
# RAPPORT DÉTAILLÉ
# ====================
print("\n" + "=" * 50)
print("RAPPORT FINAL")
print("=" * 50)

print(f"\n🎯 Modèle sélectionné : {best_model_name}")
print(f"\n📊 Métriques sur le test set :")
print(classification_report(y_test, y_pred_best, 
                          target_names=['Normal', 'Fraude']))

print(f"\n📈 Interprétation :")
print(f"   Sur {len(y_test)} transactions de test :")
print(f"   - {(y_test==0).sum()} normales")
print(f"   - {(y_test==1).sum()} frauduleuses")
print(f"\n   Le modèle a :")
print(f"   - Détecté {(y_pred_best[y_test==1]==1).sum()} fraudes sur {(y_test==1).sum()}")
print(f"   - Raté {(y_pred_best[y_test==1]==0).sum()} fraudes")
print(f"   - Créé {(y_pred_best[y_test==0]==1).sum()} fausses alertes")

print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 50)

# ====================
# INSTRUCTIONS POUR UTILISER LE MODÈLE
# ====================
print("\n📖 COMMENT UTILISER LE MODÈLE :")
print("""
1. Charger le modèle :
   import joblib
   model = joblib.load('models/fraud_detector.pkl')
   scaler = joblib.load('models/scaler.pkl')

2. Préparer une nouvelle transaction :
   new_transaction = [[V1, V2, ..., V28, Time, Amount]]
   new_transaction_scaled = scaler.transform(new_transaction)

3. Prédire :
   prediction = model.predict(new_transaction_scaled)
   # 0 = Normal, 1 = Fraude

4. Probabilité :
   probability = model.predict_proba(new_transaction_scaled)[:, 1]
   # Valeur entre 0 et 1
""")