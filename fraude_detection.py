import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Carregar dados
df = pd.read_csv('fraudes.csv')

# Exibir primeiras linhas
print("Amostra dos dados:")
print(df.head())

# Separar variáveis independentes e alvo
X = df.drop('fraude', axis=1)
y = df['fraude']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)

# Avaliação
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Importância das variáveis
importancias = modelo.feature_importances_
plt.figure(figsize=(10,6))
plt.bar(X.columns, importancias, color='purple')
plt.title('Importância das Variáveis')
plt.ylabel('Peso')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('imagens/importancia_variaveis.png')

print("\n✅ Projeto finalizado com sucesso! Gráfico salvo na pasta 'imagens'.")
