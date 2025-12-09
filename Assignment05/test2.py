# ===============================================
# Assignment 05 - Deep Feedforward Neural Networks
# Kaggle Spaceship Titanic - MLP (scikit-learn)
# 06051982
# ===============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)
from sklearn.impute import SimpleImputer

# ===============================================
# 0. Garantir que o script rode no diretório dele
# ===============================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------
# 1. Carregar dados
# -----------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Treino:", train.shape)
print("Teste:", test.shape)
print(train.head())

# -----------------------------------------------
# 2. Preparar variáveis
# -----------------------------------------------
y = train["Transported"].astype(int)
X = train.drop(["Transported"], axis=1)

# Remover Name (pouco útil)
X = X.drop(["Name"], axis=1)
test_ids = test["PassengerId"]
X_test_raw = test.drop(["Name"], axis=1)

# -----------------------------------------------
# 3. Colunas numéricas e categóricas
# -----------------------------------------------
numeric_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
categorical_features = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]

# -----------------------------------------------
# 4. Pipelines de pré-processamento
# -----------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------------------------
# 5. Dividir treino e validação
# -----------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------------
# 6. Modelo base (mais leve)
# -----------------------------------------------
mlp_base = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=50,           # menor para treinar mais rápido
    early_stopping=True,   # para de treinar quando não melhora
    n_iter_no_change=5,
    random_state=42
)

pipe_base = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp_base)
])

print("\nTreinando modelo base...")
pipe_base.fit(X_train, y_train)

y_val_pred = pipe_base.predict(X_val)
print("Acurácia validação (modelo base):", accuracy_score(y_val, y_val_pred))
print("\nRelatório de classificação (modelo base):")
print(classification_report(y_val, y_val_pred))

# -----------------------------------------------
# 7. Curva de aprendizado (versão mais rápida)
# -----------------------------------------------
print("\nGerando curva de aprendizado (pode levar um pouco, mas bem menos)...")

train_sizes, train_scores, val_scores = learning_curve(
    pipe_base,
    X, y,
    cv=3,  # antes 3 — reduz o custo
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 3),  # menos pontos
    n_jobs=-1
)

plt.figure()
plt.title("Curva de Aprendizado - Modelo Base")
plt.xlabel("Número de amostras")
plt.ylabel("Acurácia")
plt.grid()

plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino")
plt.plot(train_sizes, val_scores.mean(axis=1), "s-", label="Validação")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 8. Matriz de confusão - modelo base
# -----------------------------------------------
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matriz de Confusão - Modelo Base")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 9. GRID SEARCH - Melhor modelo (grade reduzida)
# -----------------------------------------------
print("\nIniciando Grid Search (grade reduzida para ser mais rápida)...")

mlp = MLPClassifier(
    activation="relu",
    solver="adam",
    max_iter=50,          # menor
    early_stopping=True,  # para de treinar se não melhora
    n_iter_no_change=5,
    random_state=42
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp)
])

# Grade bem menor que a original
param_grid = {
    "mlp__hidden_layer_sizes": [
        (64,), (64, 32)
    ],
    "mlp__alpha": [1e-4, 1e-3],
    "mlp__learning_rate_init": [1e-3],
    "mlp__batch_size": [64]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=2,              # antes 3
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\nMelhores parâmetros encontrados:")
print(grid.best_params_)

best_model = grid.best_estimator_

y_val_pred_best = best_model.predict(X_val)
print("Acurácia validação (melhor modelo):", accuracy_score(y_val, y_val_pred_best))
print("\nRelatório de classificação (melhor modelo):")
print(classification_report(y_val, y_val_pred_best))

# -----------------------------------------------
# 10. Treinar melhor modelo no dataset inteiro
# -----------------------------------------------
print("\nTreinando melhor modelo em TODO o conjunto de treino (X, y)...")
best_model.fit(X, y)

X_test = X_test_raw.copy()
y_test_pred = best_model.predict(X_test)
transported_pred = y_test_pred.astype(bool)

# -----------------------------------------------
# 11. Gerar submission.csv
# -----------------------------------------------
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": transported_pred
})

submission.to_csv("submission.csv", index=False)
print("\nArquivo 'submission.csv' gerado com sucesso!")
