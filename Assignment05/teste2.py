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
# 6. Modelo base (sem muita regularização)
# -----------------------------------------------
mlp_base = MLPClassifier(
    hidden_layer_sizes=(64,),   # arquitetura simples
    activation="relu",
    solver="adam",              # otimizador padrão
    alpha=1e-4,                 # L2 fraca
    learning_rate_init=1e-3,
    max_iter=50,
    early_stopping=False,       # sem early_stopping no modelo base
    random_state=42
)

pipe_base = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp_base)
])

print("\nTreinando modelo base...")
pipe_base.fit(X_train, y_train)

y_val_pred_base = pipe_base.predict(X_val)
acc_base = accuracy_score(y_val, y_val_pred_base)
print("Acurácia validação (modelo base):", acc_base)
print("\nRelatório de classificação (modelo base):")
print(classification_report(y_val, y_val_pred_base))

# -----------------------------------------------
# 7. Curva de aprendizado - modelo base
# -----------------------------------------------
print("\nGerando curva de aprendizado - Modelo Base...")

train_sizes_base, train_scores_base, val_scores_base = learning_curve(
    pipe_base,
    X, y,
    cv=3,
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1
)

plt.figure()
plt.title("Curva de Aprendizado - Modelo Base")
plt.xlabel("Número de amostras de treino")
plt.ylabel("Acurácia")
plt.grid()

plt.plot(train_sizes_base, train_scores_base.mean(axis=1), "o-", label="Treino")
plt.plot(train_sizes_base, val_scores_base.mean(axis=1), "s-", label="Validação")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 8. Matriz de confusão - modelo base
# -----------------------------------------------
cm_base = confusion_matrix(y_val, y_val_pred_base)
disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base)
disp_base.plot()
plt.title("Matriz de Confusão - Modelo Base")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 9. GRID SEARCH - Melhor modelo
#    Varia:
#      - arquitetura (camadas/neuronios)
#      - regularização L2 (alpha)
#      - early_stopping (tipo de regularização)
#      - solver (otimizador: adam x sgd)
# -----------------------------------------------
print("\nIniciando Grid Search (modelo melhorado)...")

mlp = MLPClassifier(
    activation="relu",
    learning_rate_init=1e-3,
    max_iter=80,        # um pouco maior para convergir melhor
    random_state=42
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp)
])

param_grid = {
    "mlp__hidden_layer_sizes": [
        (64,),
        (128,),
        (64, 32),
        (128, 64)
    ],
    "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],    # L2 fraca -> forte
    "mlp__early_stopping": [True, False],      # outro tipo de regularização
    "mlp__solver": ["adam", "sgd"],            # otimizadores
    "mlp__batch_size": [64]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\nMelhores parâmetros encontrados:")
print(grid.best_params_)

best_model = grid.best_estimator_
best_mlp = best_model.named_steps["mlp"]
print("\nArquitetura e parâmetros do melhor MLP:")
print("hidden_layer_sizes:", best_mlp.hidden_layer_sizes)
print("activation:", best_mlp.activation)
print("solver:", best_mlp.solver)
print("alpha (L2):", best_mlp.alpha)
print("early_stopping:", best_mlp.early_stopping)
print("batch_size:", best_mlp.batch_size)
print("learning_rate_init:", best_mlp.learning_rate_init)

# -----------------------------------------------
# 10. Avaliar melhor modelo na validação
# -----------------------------------------------
y_val_pred_best = best_model.predict(X_val)
acc_best = accuracy_score(y_val, y_val_pred_best)
print("\nAcurácia validação (melhor modelo):", acc_best)
print("\nRelatório de classificação (melhor modelo):")
print(classification_report(y_val, y_val_pred_best))

# -----------------------------------------------
# 11. Curva de aprendizado - melhor modelo
# -----------------------------------------------
print("\nGerando curva de aprendizado - Melhor Modelo...")

train_sizes_best, train_scores_best, val_scores_best = learning_curve(
    best_model,
    X, y,
    cv=3,
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1
)

plt.figure()
plt.title("Curva de Aprendizado - Melhor Modelo")
plt.xlabel("Número de amostras de treino")
plt.ylabel("Acurácia")
plt.grid()

plt.plot(train_sizes_best, train_scores_best.mean(axis=1), "o-", label="Treino")
plt.plot(train_sizes_best, val_scores_best.mean(axis=1), "s-", label="Validação")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 12. Matriz de confusão - melhor modelo
# -----------------------------------------------
cm_best = confusion_matrix(y_val, y_val_pred_best)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best)
disp_best.plot()
plt.title("Matriz de Confusão - Melhor Modelo")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 13. Treinar melhor modelo em TODO o conjunto
#     e gerar submission.csv para Kaggle
# -----------------------------------------------
print("\nTreinando melhor modelo em TODO o conjunto de treino (X, y)...")
best_model.fit(X, y)

X_test = X_test_raw.copy()
y_test_pred = best_model.predict(X_test)
transported_pred = y_test_pred.astype(bool)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": transported_pred
})

submission.to_csv("submission.csv", index=False)
print("\nArquivo 'submission.csv' gerado com sucesso!")
