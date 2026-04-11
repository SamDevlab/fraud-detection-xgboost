import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1. Carregamento dos dados
print("Carregando dados...")
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# 2. Pré-processamento
# O 'Amount' costuma ter outliers pesados; a escala logarítmica ou padronização ajuda
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
# Removemos colunas originais que não serão usadas diretamente
df = df.drop(['Time', 'Amount'], axis=1)

# 3. Divisão Treino e Teste (Obrigatório antes do SMOTE)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 4. Tratamento de Desbalanceamento (SMOTE) apenas no Treino
# Isso evita que o modelo "veja" dados sintéticos baseados no teste
print("Aplicando SMOTE para balancear as classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 5. Treinamento do Modelo (XGBoost)
print("Treinando o modelo XGBoost...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1, # Já balanceamos com SMOTE, senão usaríamos peso aqui
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)

# 6. Predições
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# 7. Avaliação Completa
print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')

# Curva Precision-Recall (A melhor para dados desbalanceados)
plt.subplot(1, 2, 2)
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, label=f'AUC-PR = {pr_auc:.4f}')
plt.title('Curva Precision-Recall')
plt.xlabel('Recall (Revocação)')
plt.ylabel('Precision (Precisão)')
plt.legend()

plt.tight_layout()
plt.show()

# 8. Importância das Variáveis
importancias = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
importancias.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title('Top 10 Variáveis mais Importantes para Detecção de Fraude')
plt.show()