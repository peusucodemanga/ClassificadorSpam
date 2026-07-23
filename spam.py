import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregamento e seleção das colunas de interesse
baseDeDados = pd.read_csv('spam.csv', encoding='cp1252')
baseDeDados = baseDeDados[['v1', 'v2']].drop_duplicates()

# 1. Quantidade de registros e atributos
shape = baseDeDados.shape

# 2. Tipos de variáveis e valores ausentes
info_df = pd.DataFrame({
    'Tipo': baseDeDados.dtypes,
    'Valores Ausentes': baseDeDados.isnull().sum(),
    'Valores Únicos': baseDeDados.nunique()
})

# 3. Duplicações
duplicados = baseDeDados.duplicated().sum()

# 4. Distribuição do Atributo Alvo
distribuicao_alvo = baseDeDados['v1'].value_counts()
proporcao_alvo = baseDeDados['v1'].value_counts(normalize=True) * 100

print(f"Dimensões do Dataset: {shape[0]} linhas e {shape[1]} colunas.\n")
print("--- Informações sobre os Atributos ---")
print(info_df)
print(f"\nQuantidade de registros duplicados: {duplicados}\n")
print("--- Distribuição do Atributo Alvo (v1) ---")
print(pd.DataFrame({'Contagem': distribuicao_alvo, 'Porcentagem (%)': proporcao_alvo}))

# Ajustando estilo dos gráficos
sns.set_theme(style="whitegrid")

# 1. Criação de atributo derivado: tamanho da mensagem em caracteres
baseDeDados['tamanho_mensagem'] = baseDeDados['v2'].apply(len)

# 2. Medidas de Localidade e Dispersão do Tamanho das Mensagens por Classe
estatisticas_tamanho = baseDeDados.groupby('v1')['tamanho_mensagem'].agg(
    Média='mean',
    Mediana='median',
    Desvio_Padrão='std',
    Mínimo='min',
    Máximo='max',
    IQR=lambda x: np.percentile(x, 75) - np.percentile(x, 25)
)

print("=== MEDIDAS DE LOCALIDADE E DISPERSÃO (Tamanho da Mensagem em Caracteres) ===")
print(estatisticas_tamanho)

# 3. Histograma Comparativo do Tamanho das Mensagens por Classe
plt.figure(figsize=(10, 5))
sns.histplot(
    data=baseDeDados,
    x='tamanho_mensagem',
    hue='v1',
    element='step',
    stat='density',
    common_norm=False,
    bins=50,
    palette={'ham': 'blue', 'spam': 'red'}
)
plt.title('Histograma da Distribuição do Tamanho das Mensagens por Classe (Normalizado)', fontsize=14, fontweight='bold')
plt.xlabel('Tamanho da Mensagem (Número de Caracteres)', fontsize=12)
plt.ylabel('Densidade', fontsize=12)
plt.legend(title='Classe', labels=['Spam (Vermelho)', 'Ham (Azul)'])
plt.tight_layout()
plt.show()

# 1. Tratamento de Inconsistências e Atributos Irrelevantes (Encoding e Filtro de Colunas)
baseDeDados = pd.read_csv('spam.csv', encoding='cp1252')
baseDeDados = baseDeDados[['v1','v2']]
xBase = baseDeDados['v2']
yBase = baseDeDados['v1']

# 2. Tratamento de Variáveis Categóricas e Valores Extremos (Vetorização)
vetor = CountVectorizer(lowercase=True, min_df=1, max_df=0.95)

xTreino, xTeste, yTreino, yTeste = train_test_split(xBase, yBase, test_size=0.2, random_state= 67)

xTreinoVetorizado = vetor.fit_transform(xTreino)
xTesteVetorizado = vetor.transform(xTeste)

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(xTreinoVetorizado, yTreino)

# Modelo 1: SGDClassifier
modelo = SGDClassifier(alpha=0.0000000001, random_state=67)
modelo.fit(xTreinoVetorizado, yTreino)

# Modelo 2: RandomForestClassifier
modelo2 = RandomForestClassifier(random_state=67)
modelo2.fit(xTreinoVetorizado, yTreino)

# Gerando as predições
pred_baseline = baseline.predict(xTesteVetorizado)
pred_treino_baseline = baseline.predict(xTreinoVetorizado)

predicoes_sgd = modelo.predict(xTesteVetorizado)
predicoes_treino_sgd = modelo.predict(xTreinoVetorizado)

predicoes_rf = modelo2.predict(xTesteVetorizado)
predicoes_treino_rf = modelo2.predict(xTreinoVetorizado)

print("="*50)
print("AVALIAÇÃO: BASELINE (Dummy)")
print("="*50)
print(f"Acurácia Teste: {accuracy_score(yTeste, pred_baseline):.4f}")
print(f"Acurácia Treino: {accuracy_score(yTreino, pred_treino_baseline):.4f}\n")
print(classification_report(yTeste, pred_baseline, zero_division=0))

print("="*50)
print("AVALIAÇÃO DO MODELO: SGDClassifier")
print("="*50)
print(f"Acurácia Teste: {accuracy_score(yTeste, predicoes_sgd):.4f}")
print(f"Acurácia Treino: {accuracy_score(yTreino, predicoes_treino_sgd):.4f}\n")
print(classification_report(yTeste, predicoes_sgd))

print("\n" + "="*50)
print("AVALIAÇÃO DO MODELO: RandomForestClassifier")
print("="*50)
print(f"Acurácia Teste: {accuracy_score(yTeste, predicoes_rf):.4f}")
print(f"Acurácia Treino: {accuracy_score(yTreino, predicoes_treino_rf):.4f}\n")
print(classification_report(yTeste, predicoes_rf))

meu_texto = ["Congratulations! You won a free prize go claim it at the nearest Amazon Shop in your Area"]
texto_vetorizado_novo = vetor.transform(meu_texto)

res_sgd = modelo.predict(texto_vetorizado_novo)
res_rf = modelo2.predict(texto_vetorizado_novo)

print("Texto Inserido:", meu_texto[0])
print("Resultado SGDClassifier:", res_sgd[0])
print("Resultado RandomForest:", res_rf[0])