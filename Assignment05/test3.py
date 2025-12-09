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
print("\nPrimeiras linhas do treino:")
print(train.head())

# -----------------------------------------------
# 1.1. Avaliar atributos e dados faltantes
# (parte pedida no enunciado)
# -----------------------------------------------
print("\nInformações do conjunto de treino:")
print(train.info())

print("\nQuantidade de valores faltantes por coluna (treino):")
print(train.isna().sum())

# -----------------------------------------------
# 2. Preparar variáveis (target e features)
# -----------------------------------------------
y = train["Transported"].astype(int)
X = train.drop(["Transported"], axis=1)

# Remover Name (pouco útil para o modelo)
if "Name" in X.columns:
    X = X.drop(["Name"], axis=1)

test_ids = test["PassengerId"]
X_test_raw = test.copy()
if "Name" in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(["Name"], axis=1)

# -----------------------------------------------
# 3. Feature Engineering
#    (melhora importante de desempenho)
# -----------------------------------------------

def split_cabin(df):
    """
    Divide a coluna Cabin em:
    - Deck  (string)
    - CabinNum (numérico)
    - Side  (string)
    """
    cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
    # Quando Cabin é NaN, str vira "nan", então tratamos isso:
    deck = cabin_split[0].replace("nan", np.nan)
    num = cabin_split[1].replace("nan", np.nan)
    side = cabin_split[2].replace("nan", np.nan)

    df["Deck"] = deck
    df["CabinNum"] = pd.to_numeric(num, errors="coerce")
    df["Side"] = side
    return df

def add_total_spent(df):
    """
    Cria a feature TotalSpent somando os gastos em diferentes serviços.
    """
    cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    # Garante que as colunas existem (por segurança)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df["TotalSpent"] = df[cols].sum(axis=1)
    return df

def add_group_id(df):
    """
    Cria a feature GroupId a partir do PassengerId (parte antes do "_").
    """
    df["GroupId"] = df["PassengerId"].astype(str).str.split("_", expand=True)[0]
    return df

# Aplicar Feature Engineering em treino e teste
X = split_cabin(X)
X = add_total_spent(X)
X = add_group_id(X)

X_test_raw = split_cabin(X_test_raw)
X_test_raw = add_total_spent(X_test_raw)
X_test_raw = add_group_id(X_test_raw)

# -----------------------------------------------
# 3.1. Definir colunas numéricas e categóricas
#     (após as novas features)
# -----------------------------------------------
numeric_features = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "CabinNum",
    "TotalSpent"
]

categorical_features = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "GroupId"
]

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
# 6. MODELO BASE (MLP simples)
# -----------------------------------------------
mlp_base = MLPClassifier(
    hidden_layer_sizes=(64,),     # 1 camada oculta com 64 neurônios
    activation="relu",
    solver="adam",                # otimizador pedido no enunciado
    alpha=1e-4,                   # regularização L2 fraca
    learning_rate_init=1e-3,
    max_iter=200,                 # mais iterações para garantir convergência
    early_stopping=True,          # para quando não melhora
    n_iter_no_change=10,
    random_state=42
)

pipe_base = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp_base)
])

print("\nTreinando MODELO BASE...")
pipe_base.fit(X_train, y_train)

y_val_pred_base = pipe_base.predict(X_val)
acc_base = accuracy_score(y_val, y_val_pred_base)

print("\n=== RESULTADOS - MODELO BASE ===")
print("Acurácia validação (modelo base):", acc_base)
print("\nRelatório de classificação (modelo base):")
print(classification_report(y_val, y_val_pred_base))

# -----------------------------------------------
# 7. Curva de aprendizado - MODELO BASE
# -----------------------------------------------
print("\nGerando curva de aprendizado - MODELO BASE (pode levar um tempo)...")

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
# 8. Matriz de confusão - MODELO BASE
# -----------------------------------------------
cm_base = confusion_matrix(y_val, y_val_pred_base)
disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base)
disp_base.plot()
plt.title("Matriz de Confusão - Modelo Base")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 9. GRID SEARCH - MODELO APRIMORADO
# -----------------------------------------------
print("\nIniciando Grid Search para MODELO APRIMORADO...")

mlp_tuned = MLPClassifier(
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=42
)

pipe_tuned = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp_tuned)
])

# Grade reduzida, mas ainda explorando capacidade + regularização
param_grid = {
    "mlp__hidden_layer_sizes": [
        (64,),        # igual ao base
        (128,),       # mais neurônios
        (128, 64)     # 2 camadas ocultas
    ],
    "mlp__alpha": [1e-4, 1e-3, 1e-2],      # diferentes níveis de regularização
    "mlp__learning_rate_init": [1e-3, 5e-4],
    "mlp__batch_size": [64, 128]
}

grid = GridSearchCV(
    pipe_tuned,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("\nMelhores parâmetros encontrados (MODELO APRIMORADO):")
print(grid.best_params_)

best_model = grid.best_estimator_

y_val_pred_best = best_model.predict(X_val)
acc_best = accuracy_score(y_val, y_val_pred_best)

print("\n=== RESULTADOS - MODELO APRIMORADO ===")
print("Acurácia validação (melhor modelo):", acc_best)
print("\nRelatório de classificação (melhor modelo):")
print(classification_report(y_val, y_val_pred_best))

# -----------------------------------------------
# 10. Curva de aprendizado - MODELO APRIMORADO
# -----------------------------------------------
print("\nGerando curva de aprendizado - MODELO APRIMORADO (pode levar um tempo)...")

train_sizes_best, train_scores_best, val_scores_best = learning_curve(
    best_model,
    X, y,
    cv=3,
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1
)

plt.figure()
plt.title("Curva de Aprendizado - Modelo Aprimorado")
plt.xlabel("Número de amostras de treino")
plt.ylabel("Acurácia")
plt.grid()

plt.plot(train_sizes_best, train_scores_best.mean(axis=1), "o-", label="Treino")
plt.plot(train_sizes_best, val_scores_best.mean(axis=1), "s-", label="Validação")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 11. Matriz de confusão - MODELO APRIMORADO
# -----------------------------------------------
cm_best = confusion_matrix(y_val, y_val_pred_best)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best)
disp_best.plot()
plt.title("Matriz de Confusão - Modelo Aprimorado")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 12. Treinar melhor modelo no dataset inteiro
# -----------------------------------------------
print("\nTreinando MELHOR MODELO em TODO o conjunto de treino (X, y)...")
best_model.fit(X, y)

X_test = X_test_raw.copy()
y_test_pred = best_model.predict(X_test)
transported_pred = y_test_pred.astype(bool)

# -----------------------------------------------
# 13. Gerar submission.csv
# -----------------------------------------------
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": transported_pred
})

submission.to_csv("submission.csv", index=False)
print("\nArquivo 'submission.csv' gerado com sucesso!")
print("Use esse arquivo para submissão no Kaggle (Spaceship Titanic).")
