'''
Este programa está muito bem estruturado para resolver uma tarefa de regressão supervisionada com o
algoritmo XGBoost, que é altamente eficaz e usado em competições de Machine Learning.

✅ Pontos fortes:

Realiza todo o pipeline: leitura, limpeza, codificação, divisão dos dados, treinamento, avaliação e
interpretação.

Usa o LabelEncoder  para lidar com variáveis categóricas automaticamente.

Avalia o modelo com métricas apropriadas (RMSE e R²).

Apresenta visualmente a importância das features — útil para análise interpretativa.


⚠️ Sugestões de melhorias opcionais:

Normalização de variáveis: pode melhorar a performance, principalmente se adicionar outros algoritmos
além do XGBoost.

Validação cruzada (cross-validation): para obter resultados mais robustos e evitar overfitting.

GridSearch ou RandomizedSearch: para ajustar hiperparâmetros do XGBoost.

Salvar o modelo treinado com joblib ou pickle se for usar em produção.

'''

# Importa bibliotecas essenciais
import pandas as pd                              # Para manipulação de dados tabulares
import xgboost as xgb                            # Algoritmo XGBoost para regressão
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from sklearn.metrics import mean_squared_error, r2_score  # Para avaliar a performance do modelo
from sklearn.preprocessing import LabelEncoder   # Para transformar variáveis categóricas em números
import matplotlib.pyplot as plt                  # Para visualizar a importância das variáveis
import numpy as np                               # Para operações numéricas        

# Lê o dataset do arquivo CSV
df = pd.read_csv("teen_phone_addiction_dataset.csv")  # Certifique-se de que o arquivo esteja no mesmo diretório

# Remove colunas que não contribuem para o modelo (como ID e Name)
df.drop(columns=['ID', 'Name'], inplace=True)

# Identifica automaticamente as colunas do tipo "object" (texto/categóricas)
categorical_cols = df.select_dtypes(include='object').columns

# Converte cada coluna categórica em valores numéricos usando LabelEncoder
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Garante que todos os valores sejam tratados como string

# Remove qualquer linha que contenha valores ausentes (NaN)
df.dropna(inplace=True)

# Define a variável alvo que queremos prever
target = 'Addiction_Level'

# Define a lista de variáveis independentes (todas as colunas exceto a variável alvo)
features = [col for col in df.columns if col != target]

# Cria os conjuntos de entrada (X) e saída (y)
X = df[features]
y = df[target]

# Separa os dados em treino (80%) e teste (20%) de forma aleatória, com uma semente fixa para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria o modelo de regressão XGBoost com objetivo de minimizar o erro quadrático
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Treina o modelo com os dados de treino
model.fit(X_train, y_train)

# Realiza previsões com os dados de teste
y_pred = model.predict(X_test)

# Calcula o Erro Médio Quadrático (RMSE) - quanto menor, melhor
#rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calcula o coeficiente de determinação (R²) - quanto mais próximo de 1, melhor
r2 = r2_score(y_test, y_pred)

# Exibe as métricas de avaliação do modelo
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Gera um gráfico com a importância relativa de cada variável preditora
xgb.plot_importance(model)
plt.title("Importância das variáveis")
plt.show()

