# teen_phone_addiction
'''
Descrição Geral da Base de Dados: teen_phone_addiction_dataset.csv
Essa base parece estar relacionada ao nível de vício em celular entre adolescentes. 
Foi construída a partir de questionários, entrevistas ou registros comportamentais de jovens, 
com o objetivo de prever o grau de dependência do celular com base em diversos fatores.

🔍 Estrutura esperada da base de dados

✅ Colunas que foram removidas:
ID: identificador único de cada adolescente — sem utilidade para predição.

Name: nome da pessoa — também irrelevante para o modelo.

🎯 Variável alvo (target):

Addiction_Level: nível de dependência do celular (provavelmente numérico e contínuo).
Ex: pode ir de 0 a 10, onde 0 = sem dependência, 10 = alto vício.

📥 Variáveis preditoras (features):

O restante das colunas foi usado para prever o Addiction_Level. Devem incluir:

Tipo de variável	Exemplos possíveis

Categóricas	        Gênero, Tipo de escola, Ocupação dos pais
Numéricas	        Horas no celular por dia, notas escolares, idade
Comportamentais	    Frequência de uso de redes sociais, jogos, etc.

Essas variáveis foram:

Codificadas com LabelEncoder (as categóricas).

Normalizadas com StandardScaler (as numéricas).

Filtradas por outliers usando a técnica do IQR.


Este programa está estruturado para resolver uma tarefa de regressão supervisionada com o
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
