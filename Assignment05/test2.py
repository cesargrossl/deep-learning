# ===============================================
# Assignment 05 - Deep Feedforward Neural Networks
# Kaggle Spaceship Titanic - MLP (scikit-learn)
# 06051982
# ===============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
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
# 0) Garantir que o script rode no diretório dele
# ===============================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

RANDOM_STATE = 42


# ===============================================
# Funções auxiliares (curva de aprendizado / confusão)
# ===============================================
def plot_learning_curve(estimator, X, y, title, cv=3):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X, y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )

    plt.figure()
    plt.title(title)
    plt.xlabel("Número de amostras")
    plt.ylabel("Acurácia")
    plt.grid()

    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Treino")
    plt.plot(train_sizes, val_scores.mean(axis=1), "s-", label="Validação")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_model_summary(pipe, label):
    mlp = pipe.named_steps["mlp"]
    print(f"\n===== {label} | Arquitetura e Parâmetros (MLP) =====")
    print(mlp)  # imprime a “arquitetura” (camadas, etc.)
    print("hidden_layer_sizes:", mlp.hidden_layer_sizes)
    print("activation:", mlp.activation)
    print("solver:", mlp.solver)
    print("alpha (L2):", mlp.alpha)
    print("learning_rate_init:", mlp.learning_rate_init)
    print("batch_size:", mlp.batch_size)
    print("max_iter:", mlp.max_iter)
    print("early_stopping:", mlp.early_stopping)
    if hasattr(mlp, "momentum"):
        print("momentum:", mlp.momentum)
    if hasattr(mlp, "learning_rate"):
        print("learning_rate:", mlp.learning_rate)


# ===============================================
# 1) Carregar dados
# ===============================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Treino:", train.shape)
print("Teste:", test.shape)
print(train.head())

# ===============================================
# 2) Preparar variáveis
# ===============================================
y = train["Transported"].astype(int)
X = train.drop(["Transported"], axis=1)

# Remover Name (geralmente pouco útil)
if "Name" in X.columns:
    X = X.drop(["Name"], axis=1)

test_ids = test["PassengerId"].copy()
X_test_raw = test.drop(["Name"], axis=1) if "Name" in test.columns else test.copy()

# ===============================================
# 3) Colunas numéricas e categóricas
# ===============================================
numeric_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
categorical_features = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]

# ===============================================
# 4) Pipelines de pré-processamento
# ===============================================
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

# ===============================================
# 5) Dividir treino e validação
# ===============================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ===============================================
# 6) MODELO BASE
# ===============================================
mlp_base = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    alpha=1e-4,               # L2 (regularização)
    learning_rate_init=1e-3,
    batch_size=64,
    max_iter=150,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=RANDOM_STATE
)

pipe_base = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp_base)
])

print("\n===============================")
print("Treinando MODELO BASE...")
print("===============================")

pipe_base.fit(X_train, y_train)
y_val_pred_base = pipe_base.predict(X_val)

acc_base = accuracy_score(y_val, y_val_pred_base)
print("Acurácia validação (BASE):", acc_base)
print("\nRelatório de classificação (BASE):")
print(classification_report(y_val, y_val_pred_base))

plot_learning_curve(pipe_base, X, y, "Curva de Aprendizado - Modelo Base", cv=3)
plot_conf_matrix(y_val, y_val_pred_base, "Matriz de Confusão - Modelo Base")

# Pequena análise (como o enunciado pede) — sem “relatório”, apenas output no código
print("\nANÁLISE (BASE):")
print("- O modelo base utiliza uma única camada oculta, tendendo a ter menor capacidade de representação.")
print("- A curva de aprendizado permite observar se há subajuste (treino e validação baixos) ou sobreajuste (treino alto, validação menor).")


# ===============================================
# 7) MODELO MELHORADO 1 (ADAM) - GRID SEARCH
#    Ajuste de arquitetura + L2 + LR
# ===============================================
print("\n===============================================")
print("GridSearch - MODELO MELHORADO 1 (ADAM)...")
print("===============================================")

mlp_adam = MLPClassifier(
    solver="adam",
    activation="relu",
    early_stopping=True,
    max_iter=250,
    n_iter_no_change=10,
    random_state=RANDOM_STATE
)

pipe_adam = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp_adam)
])

param_grid_adam = {
    "mlp__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
    "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],            # L2 (regularização)
    "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3],
    "mlp__batch_size": [32, 64, 128]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

grid_adam = GridSearchCV(
    pipe_adam,
    param_grid_adam,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid_adam.fit(X_train, y_train)
best_adam = grid_adam.best_estimator_

print("\nMelhores parâmetros (ADAM):")
print(grid_adam.best_params_)
print("Melhor score CV (ADAM):", grid_adam.best_score_)

y_val_pred_adam = best_adam.predict(X_val)
acc_adam = accuracy_score(y_val, y_val_pred_adam)

print("\nAcurácia validação (MELHORADO 1 - ADAM):", acc_adam)
print("\nRelatório de classificação (MELHORADO 1 - ADAM):")
print(classification_report(y_val, y_val_pred_adam))

print_model_summary(best_adam, "MELHORADO 1 - ADAM")

plot_learning_curve(best_adam, X, y, "Curva de Aprendizado - Melhorado 1 (ADAM)", cv=3)
plot_conf_matrix(y_val, y_val_pred_adam, "Matriz de Confusão - Melhorado 1 (ADAM)")

print("\nANÁLISE (MELHORADO 1 - ADAM):")
print("- A variação de hidden_layer_sizes altera a capacidade do modelo (mais neurônios/camadas => maior capacidade).")
print("- O alpha controla regularização L2: valores maiores tendem a reduzir overfitting, mas podem reduzir a acurácia de treino.")
print("- A learning_rate_init influencia a convergência: taxa muito alta pode instabilizar; muito baixa pode não convergir bem.")


# ===============================================
# 8) MODELO MELHORADO 2 (SGD) - GRID SEARCH
#    Otimizador diferente (como o enunciado sugere)
# ===============================================
print("\n===============================================")
print("GridSearch - MODELO MELHORADO 2 (SGD)...")
print("===============================================")

mlp_sgd = MLPClassifier(
    solver="sgd",
    activation="relu",
    early_stopping=True,
    max_iter=300,
    n_iter_no_change=10,
    random_state=RANDOM_STATE
)

pipe_sgd = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("mlp", mlp_sgd)
])

param_grid_sgd = {
    "mlp__hidden_layer_sizes": [(64,), (128,), (128, 64)],
    "mlp__alpha": [1e-5, 1e-4, 1e-3],                 # L2 (regularização)
    "mlp__learning_rate": ["constant", "adaptive"],
    "mlp__learning_rate_init": [1e-3, 5e-3, 1e-2],
    "mlp__momentum": [0.0, 0.9],
    "mlp__batch_size": [32, 64, 128]
}

grid_sgd = GridSearchCV(
    pipe_sgd,
    param_grid_sgd,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid_sgd.fit(X_train, y_train)
best_sgd = grid_sgd.best_estimator_

print("\nMelhores parâmetros (SGD):")
print(grid_sgd.best_params_)
print("Melhor score CV (SGD):", grid_sgd.best_score_)

y_val_pred_sgd = best_sgd.predict(X_val)
acc_sgd = accuracy_score(y_val, y_val_pred_sgd)

print("\nAcurácia validação (MELHORADO 2 - SGD):", acc_sgd)
print("\nRelatório de classificação (MELHORADO 2 - SGD):")
print(classification_report(y_val, y_val_pred_sgd))

print_model_summary(best_sgd, "MELHORADO 2 - SGD")

plot_learning_curve(best_sgd, X, y, "Curva de Aprendizado - Melhorado 2 (SGD)", cv=3)
plot_conf_matrix(y_val, y_val_pred_sgd, "Matriz de Confusão - Melhorado 2 (SGD)")

print("\nANÁLISE (MELHORADO 2 - SGD):")
print("- A troca do otimizador (ADAM -> SGD) altera a dinâmica de treinamento.")
print("- SGD pode exigir ajuste fino de learning rate e momentum; por isso a grade inclui 'adaptive' e momentum.")
print("- A comparação entre ADAM e SGD evidencia o impacto do otimizador na generalização e na estabilidade.")


# ===============================================
# 9) Resumo comparativo (impacto das variações)
# ===============================================
print("\n===============================================")
print("RESUMO COMPARATIVO (Validação)")
print("===============================================")
print(f"BASE               | acc: {acc_base:.4f}")
print(f"MELHORADO 1 (ADAM) | acc: {acc_adam:.4f} | best_cv: {grid_adam.best_score_:.4f}")
print(f"MELHORADO 2 (SGD)  | acc: {acc_sgd:.4f} | best_cv: {grid_sgd.best_score_:.4f}")

print("\nINTERPRETAÇÃO (impactos):")
print("- Se a acurácia de validação sobe e o gap treino/val diminui na curva, houve melhora de generalização.")
print("- Se a acurácia de treino sobe muito e validação não acompanha, houve tendência a overfitting (regularização/alpha ajuda).")
print("- Arquiteturas maiores aumentam capacidade; regularização e early stopping ajudam a controlar sobreajuste.")
print("- Otimizadores diferentes podem mudar desempenho e estabilidade; ADAM costuma convergir mais facilmente, SGD pode exigir tuning.")


# ===============================================
# 10) Escolher o melhor (pela validação) e treinar no dataset inteiro
# ===============================================
candidates = [
    ("BASE", pipe_base, acc_base),
    ("ADAM", best_adam, acc_adam),
    ("SGD", best_sgd, acc_sgd),
]
best_name, best_pipe, best_acc = sorted(candidates, key=lambda x: x[2], reverse=True)[0]

print("\n===============================================")
print(f"Treinando MELHOR PIPE no conjunto inteiro: {best_name} (val acc = {best_acc:.4f})")
print("===============================================")

best_pipe.fit(X, y)

# ===============================================
# 11) Prever no test e gerar submission.csv
# ===============================================
y_test_pred = best_pipe.predict(X_test_raw)
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": y_test_pred.astype(bool)
})
submission.to_csv("submission.csv", index=False)
print("\nArquivo 'submission.csv' gerado com sucesso!")
print("Modelo usado para submission:", best_name)
