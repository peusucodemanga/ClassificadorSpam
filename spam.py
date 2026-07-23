import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Pegando a base de dados em csv que não está como UTF-8 por isso do encoding cp1252 e deixando só as colunas importantes
baseDeDados = pd.read_csv('spam.csv', encoding='cp1252')
baseDeDados = baseDeDados[['v1','v2']]

# O x da base de dados será oque utilizaremos para prever no caso os textos para identificar se é spam ou não
xBase = baseDeDados['v2']

# O y da base de dados é oque estamos tentando prever no caso se é spam ou não
yBase = baseDeDados['v1']

# Aqui estamos dividindo a base em treino e teste com o tamanhho do teste sendo 20% do tamanho da base
# O random state funcionará para a gente conseguir reproduzir os resultados enquanto o número se mantiver o mesmo no caso a divisão será sempre essa no caso 67
xTreino, xTeste, yTreino, yTeste = train_test_split(xBase, yBase, test_size=0.2, random_state= 67)

# UTILIZANDO  Gradiente Descendente Estocástico

# Gerando um vetor vazio sem nenhum vocabulário por enquanto mas ele será a base que armazenará as palavras
# O CountVectorizer faz: Cria um vocabulário com todas as palavras encontradas
# Parâmetros que podem ser usados no CountVectorizer:
# lowercase= (True|False) escolher se for True ele deixa tudo em lowercase logo palavras em maiúsculo e minúsculo serão iguais se for false ele conta palavras maiúsculas como diferentes
# min_df =(int) você da um inteiro que define em quantos documentos uma palavra precisa aparecer para ser adicionada ao vocabulário
# max_df= (float) você da um float indicando em quantos documentos percentualmente uma palavra pode ser para ser adicionada não adicionaria palavras muito repetitivas.
vetor = CountVectorizer(lowercase=True, min_df=1, max_df=0.95)

# O fit_transform é um método que chama um fit seguido de um transform
# Oque o fit faz: Ele irá passar o vocabulário que será analisado dentro de seu parâmetro
# Oque o transform faz: Ele irá trasformar agora cada sentença/cadeia em um vetor contendo o número de palavras que ele possui do vocabulário.
xTreinoVetorizado = vetor.fit_transform(xTreino)
xTesteVetorizado = vetor.transform(xTeste)

# Aqui é criado o modelo do SGDClassifier
# Ele calcula a probabilidade de uma mensagem ser spam ou não, a partir dos pesos que ele atribui a cada palavra
# Fórmula é algo similar a (somatório de (xi * wi)) + b sendo xi o número de vezes que a palavra apareceu na frase lida, wi sendo o peso atribuído pelo modelo aquela palavra e b sendo o viés. 
# Em uma sentença com várias palavras ele faz Numero de vezes que a palavra apareceu * peso e faz isso para cada palavra e para indicar a qual classe ela pertence ela verifica o resultado do cálculo e analisa se está dentro do limiar de uma das classes.
# Esse alpha=0.0000000001 é responsável pela intensidade de regularização ele é utilizado para durante o treinamento controlar a penalidade
modelo =SGDClassifier(alpha=0.0000000001, random_state=67)

# Aqui no modelo.fit é onde ele usa  irá fazer os cálculos das probabilidades tando ser ham ou spam quanto de cada palavra pertencer a ham ou spam e irá armazenálas
modelo.fit(xTreinoVetorizado, yTreino)

# Aqui no modelo.predict é onde ele faz as predições pegando as sentenças dos testes vetorizadas e tentando aplicar elas nas probabilidades tanto pra ham quanto pra spam e escolhendo a mais provável
predicoes = modelo.predict(xTesteVetorizado)
predicoes_treino = modelo.predict(xTreinoVetorizado)

# O accuracy_score serve para calcular a porcentagem de acertos que o modelo teve ele compara a tabela que diz se era ham ou spam as predições dadas no caso o retorn foi de aproximadamente 97,94% de acurácia
print("Acuracia:", accuracy_score(yTeste, predicoes))
print("Acuracia Treino:", accuracy_score(yTreino, predicoes_treino))

# TESTES A MÃO

# Aqui é um texto para testes
meu_texto = ["Congratulations! You won a free prize go claim it at the nearest Amazon Shop in your Area"]

# Aqui é feita a vetorização do texto para testes
texto_vetorizado = vetor.transform(meu_texto)

# Aqui é feita a previsão do modelo para o texto fornecido
resultado = modelo.predict(texto_vetorizado)

print(resultado)