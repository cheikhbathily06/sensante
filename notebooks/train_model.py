import pandas as pd
import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# Charger le dataset
# =========================
df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients , {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# =========================
# Encodage variables
# =========================
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# =========================
# Features & cible
# =========================
feature_cols = [
    'age',
    'sexe_encoded',
    'temperature',
    'tension_sys',
    'toux',
    'fatigue',
    'maux_tete',
    'frissons',
    'nausee',
    'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")
print(f"Cible : {y.shape}")

# =========================
# Split dataset
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEntrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# =========================
# Modèle
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# =========================
# Prédiction
# =========================
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})

print("\n", comparison)

# =========================
# Matrice de confusion
# =========================
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("\nMatrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# =========================
# Création dossier figures (FIX ERREUR)
# =========================
os.makedirs("figures", exist_ok=True)

# =========================
# Visualisation
# =========================
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()

plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()

print("\nFigure sauvegardee dans figures/confusion_matrix.png")

# =========================
# Sauvegarde modèle
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")

print("\nModele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# =========================
# Reload test
# =========================
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"\nModele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# =========================
# Test patient
# =========================
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'frissons': 1,
    'nausee': 0,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    nouveau_patient['frissons'],
    nouveau_patient['nausee'],
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]

print("\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Confiance : {probas.max():.1%}")

print("\nProbabilites :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:10s} : {proba:.1%} {bar}")