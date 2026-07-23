# Classificador de Mensagens Spam

## 🎯 Título e Objetivo
**Título:** Detecção de Spam em Mensagens de Texto
**Objetivo:** Desenvolver e comparar modelos de Machine Learning capazes de analisar mensagens de texto (SMS/E-mails) e classificá-las automaticamente como "Spam" (propagandas indesejadas/lixo eletrônico) ou mensagens legítimas.

## 👥 Integrantes
* Caio Mendonça Barreto de Brito
* Dimitri Martins Oliveira
* João Pedro Ferreira da Cruz

## 🗄️ Fonte dos Dados
A base de dados utilizada é o arquivo `spam.csv`, originalmente disponibilizada no [Kaggle](https://www.kaggle.com/code/karanbassan/spam-detection). Os dados foram importados utilizando o *encoding* `cp1252` para evitar erros de leitura e, em seguida, a base foi filtrada para manter apenas a coluna das "labels" (se é spam ou não) e a coluna das "features" (o texto bruto da mensagem).

## 🧠 Tipo da Tarefa
Trata-se de uma tarefa de **Classificação Binária** através de **Processamento de Linguagem Natural (PLN)**, utilizando aprendizado de máquina supervisionado.

## 📂 Organização dos Arquivos
O repositório/entrega está organizado da seguinte forma:
* `spam_Caio_Dimitri_JoaoPedro.ipynb`: Notebook principal contendo todo o código-fonte, do pré-processamento à predição.
* `spam.csv`: Arquivo de dados necessário para o treinamento e teste.
* `README.md`: Este arquivo de documentação.
* `spam.py`: Arquivo feito em Python inicialmente.

## 🚀 Instruções para abrir o notebook no Colab
1. Acesse o [Google Colaboratory](https://colab.research.google.com/).
2. Clique em **Arquivo > Fazer upload de notebook** e selecione o arquivo `spam_Caio_Dimitri_JoaoPedro.ipynb`.
3. No menu lateral esquerdo do Colab, clique no ícone de **Arquivos** (em formato de pasta).
4. Faça o upload do arquivo `spam.csv` para o armazenamento da sessão (ambiente de execução).
5. Após o upload, você pode rodar as células do notebook sequencialmente clicando no botão "Play" ou usando o atalho `Shift + Enter`.

## 🤖 Modelos Utilizados
Para que os algoritmos conseguissem ler as mensagens, utilizamos o **CountVectorizer** para transformar os textos em vetores numéricos de acordo com o vocabulário. Em seguida, estruturamos três modelos:
1. **Baseline (DummyClassifier):** Modelo de referência configurado com a estratégia `most_frequent`, que atua como um "chute ingênuo" prevendo sempre a classe majoritária (`ham`).
2. **SGDClassifier (Gradiente Descendente Estocástico):** Calcula a probabilidade da classe a partir de pesos atribuídos a cada palavra, controlando a penalidade no treinamento através de um fator de regularização (alpha = 1e-10).
3. **RandomForestClassifier (Floresta Aleatória):** Algoritmo baseado em *ensemble* e árvores de decisão que classifica os dados a partir do aprendizado combinado de múltiplas árvores.

## 📊 Principais Resultados
Os dados foram divididos em 80% para treino e 20% para teste (`random_state=67`). Nos dados de teste, os resultados obtidos foram:
* **Acurácia Baseline (Dummy):** ~84,84%
* **Acurácia SGDClassifier:** ~98,12%
* **Acurácia RandomForestClassifier:** ~96,68%
* *Conclusão:* Embora o SGDClassifier tenha obtido uma leve superioridade na acurácia geral, o grupo optou pelo **RandomForestClassifier** como modelo final. Em sistemas antispam, prioriza-se uma altíssima precisão (o RandomForest obteve 100% de precisão para a classe spam no teste), evitando falsos positivos que poderiam bloquear mensagens legítimas urgentes.

## 🤝 Divisão das Contribuições
* **Caio:**  Confecção do código (Tratamento inicial do CSV) e comentários explicativos sobre ele.
* **Dimitri:** Confecção do código (Treinamento e teste do modelo de Classificação SGDClassifier) e divisão do vídeo e falas.
* **João Pedro:** Confecção do código (Treinamento e teste do modelo de Classificação RandomForestClassifier) e edição do vídeo.

## 🎥 Link do Vídeo
[Link clicável aqui!](https://youtu.be/Uv2gI1iUogE)

## 🤖 Declaração de Uso de Ferramentas de Inteligência Artificial
Declaramos que ferramentas de Inteligência Artificial (como Gemini/ChatGPT) foram utilizadas pontualmente neste projeto para auxiliar na estruturação e revisão textual deste arquivo `README.md`, na formatação do notebook `spam_Caio_Dimitri_JoaoPedro.ipynb` e na divisão do roteiro de apresentação. A lógica de programação principal, o desenvolvimento das premissas dos algoritmos e a execução dos testes foram conduzidas de forma autoral pelo grupo.