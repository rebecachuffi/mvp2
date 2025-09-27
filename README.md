# **MVP2 - Fraude em Pagamentos**
Machine Leaning e Analytics PUC - Rio (2025)


**Nome:** Rebeca Chuffi Saccochi


**Duração:** 30 horas 


[Notebook](https://github.com/rebecachuffi/mvp2/blob/main/V2_MVP2_RebecaChuffi.ipynb) | [Repositório](https://github.com/rebecachuffi/mvp2/tree/main)

## 1 | Escopo, objetivo e definição do problema
------------

### **Contexto**
Este projeto é o segundo na área de Ciência de Dados da minha vida! Desta vez escolhi um dataset maior para entender os desafios de tratar e processar muitos dados. O treinamento é mais lento, a otimização de parâmetros também. Notei que não é possível testar todas as combinações de modelos possíveis, então o conhecimento prévio dos modelos se faz muito mais necessário.   

### Descrição do Problema
O crescimento acelerado do comércio eletrônico e dos pagamentos digitais nos últimos anos fez com que tivéssemos um volume muito maior de dados à disposição (o que é parte essencial do treinamento de um modelo) e o surgimentos de novas necessidades em termos de detecção de possíveis fraudes. Em 2023, um [estudos](https://thepaypers.com/fraud-and-fincrime/thought-leader-insights/trends-in-the-fraud-and-payments-space-for-2023?utm_source=chatgpt.com) mostrou que as perdas globais com fraudes de pagamento online ultrapassam os USD 48 bilhões. Comparativamente, foram registrados USD 41 bilhões de perdas em fraudes 2022.   

Além disso, os tipos de fraudes não consistem mais apenas nos métodos tradicionais. Agora temos esquemas mais sofisticados que usam Inteligência Artificial para criar ataques mais convincentes e personalizados para cada cliente específico. Com isso, surge a necessidade de modelos mais sofisticados de detecção de tais tentativas de fraude.

O projeto tem o `objetivo` de utilizar **machine learning** para desenvolver um modelo que detecte fraudes em transações financeiras a partir dos dados disponíveis no dataset. O modelo será treinado para classificar corretamente em "transação fraudulenta" e "transação não fraudulenta" com o intuito de automatizar esse processo (visto que métodos tradicionais não são mais tão eficientes pois as ténicas de fraude evoluiram).

### Tipo de Tarefa

Esse é um problema clássico de `CLASSIFICAÇÃO`. Vamos utilizar os dados do dataset para prever dois possíveis tipos de classe:

*   Fraudulento(1)
*   Não frauduLento (0)

### Área de aplicação e valor para o negócio

No caso particular desse projeto, a **área de aplicação** gira em torno de análise de dados tabulares disponíveis no dataset de bancos e organizações financeiras. A análise será feita processando e entendendo um grande volume de dados tabulares (estruturada).

O modelo proposto poderá ser aplicado na área de segurança de pagamentos online. As técnicas utilizadas tem potencial para serem utilizada em sistemas de monitoramente online. Um segundo passo para isso (o que vai além do escopo desse projeto) seria fazer com que o se auto calibrasse em caso de *concept drift*.

O valor que esse modelo poderia gerar para o negócio inclui:
*   Redução de perdas financeiras: detecção de fraudes em tempo real.
*   Aumento da confiança dos clientes: aumenta a confiança dos consumidores, incentivando mais compras e fidelização.
*   Redução de operações de chargeback e reembolsos: Minimiza os custos associados a estornos e reembolsos, aumentando a margem de lucro.
*   Escalabilidade: podendo lidar com volumes maiores de dados conforme o negócio cresce.

━━━━━━━━━━━━━━━━━━━━

##2 | Dados: carga, entendimento e qualidade
------------

### Seleção de Dados

O [dataset](https://www.kaggle.com/datasets/sanskar457/fraud-transaction-detection/data) utilizado tem origem no link do Kaggle. Ele descreve um conjunto de dados rotulados que serão utilizados tanto para treinamento como para teste do modelo. Os atributos são:

1.   `TRANSACTION_ID:` chave primária da transação.
2.   `TX_DATETIME:` data e hora em que a transação ocorreu (nome que aqui temos o formato YYYY-MM-DD).
3.   `CUSTOMER_ID:` identificador único do cliente que realizou a transação.
4.   `TERMINAL_ID:` identificador único do terminal pelo qual a transação foi feita.
5.   `TX_AMOUNT:`quantidade de dinheiro movimentado na transação.
6.   `TX_TIME_SECONDS:`quantidade de segundos que se passaram desde o momento em que o cliente iniciou o processo de pagamento até o momento da transação ser registrada
6.   `TX_TIME_DAYS:`quantidade de dias que se passaram desde o momento em que o cliente iniciou o processo de pagamento até o momento da transação ser registrada
6.   `TX_FRAUD:` variável binária, com valor 0 para transação legítima e valor 1 para transação fraudulenta (nesse caso o "sucesso", considerando probabilidade, é encontrar a informação fraudulenta)
6.   `TX_FRAUD_SCENARIO:` variável com quatro possíveis valores, um indicando nenhuma fraude (0) e os outros três identificando dois tipos de fraudes diferentes (aqui não especificados).   



### **Tecnologias Utilizadas**

- **Python**: Para análise de dados e visualização.
- **Pandas**: Manipulação e análise de dados.
- **Matplotlib / Seaborn**: Visualização de dados.
- **Scikit-learn**: Construção e avaliação de modelos preditivos.
- **Pycountry**: Para normalização dos nomes dos países.
- **Console**: Para painel personalizado.
- **Google Colab**: Para execução de notebooks e integração com o Google Drive.
- **Rich:** Para criar interfaces de usuário no terminal.
- **Missingno:** Para visualização de dados faltantes.
- **Seaborn / Matplotlib:** Para visualização de dados.
- **NumPy / Pandas:** Manipulação e análise de dados.
- **SciPy:** Para cálculos científicos e estatísticos.
- **Scikit-learn:** Construção e avaliação de modelos preditivos.
- **Joblib:** Para salvar modelos e objetos Python.
- **IPython.display:** Para exibição de imagens em notebooks.
- **Feature Selection (SelectKBest, RFE, ExtraTreesClassifier):** Para otimização de características.
- **GridSearchCV / RandomizedSearchCV:** Para otimização de hiperparâmetros.
- **Scikit-learn Metrics:** Para avaliação de desempenho de modelos.
- 
### **Instalação**

Para executar este projeto localmente, basta abrir com Colab: [Notebook]([https://github.com/rebecachuffi/mvp1/blob/main/MVP1_RebecaChuffi.ipynb](https://github.com/rebecachuffi/mvp2/blob/main/V2_MVP2_RebecaChuffi.ipynb)) 

### **Bibliotecas**
```
# Definindo a semente para reprodutibilidade
SEED = 1811
np.random.seed(SEED)
random.seed(SEED)

# Para frameworks que suportam seed adicional (ex.: PyTorch/TensorFlow), documente aqui:
# import torch; torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
# import tensorflow as tf; tf.random.set_seed(SEED)

print("Python:", sys.version.split()[0])
print("Seed global:", SEED)

#Bibliotecas principais
import os, random, time, sys, math
import numpy as np
import pandas as pd
import time

#Visualização de Dados
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as ms # Visualização de dados faltantes

#Bibliotecas do Scikit-learn (Pré-processamento, Modelos e Métricas)
# Pré-processamento e Transformações
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Modelos
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV

# Métricas
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score,
                             silhouette_score, recall_score, classification_report)

# Seleção de Features
from sklearn.feature_selection import SelectKBest, f_classif, RFE

#Outras Bibliotecas
import gdown
import joblib
from IPython.display import Image

#Para Interatividade no Terminal
from rich.console import Console # Painel personalizado
from rich.panel import Panel     # Painel personalizado
console = Console()             # Criando o painel

#Avisos
import warnings
warnings.filterwarnings("ignore")

#Conectando ao Google Drive
from google.colab import drive  # Para fazer upload de arquivos no Google Colab
```


## 3 | Passos para determinação do modelo

Não foi necessário tanto esforço para a **limpeza de dados** para esse dataset, dito que que já estava prativamente pronto para uso (porém a real condição do dataset foi verificada para não termos possíveis erros no modelo). Métodos de feature selection aprendidos foram aplicados e foi considerado o desbalanceamento das classes em todos os processos. Além disso, os dados foram padronizados para que o modelo não desse maior força aos dados que estavam numa escala maior. 

Através de **análise exploratória de dados** e com ajuda de gráficos para nos ajudar a visualizar características importantes dos dados, pudemos ter uma noção de quais eram relevantes (o que foi avaliado posteriormente na etapa de feature selection).  Além disso, criamos um heatmap de correlação para visualizar melhor as relações entre as variáveis. 

Após o método de **feature selection** (SelectKBest e Regressão Logística) definimos 4 variáveis que teriam um peso importante na definição da variável target. Claro que, caso o modelo final não apresentasse um resultado satisfatório, teríamos que voltar nessa etapa para reavaliar essa escolha. 

O dataset foi divido em vtreino, validação e teste** e os três conjuntos tinham a mesma porcentagem de classes (ou seja, consideravam o desbalanceamento das mesmas). 

Sobre as **métricas de avaliação** como nosso problema (que envolve detecção de fraudes usando classificação binária), temos o seguinte contexto:

*   TP (True Positive): Fraudes corretamente identificadas.
*   FP (False Positive): Transações legítimas incorretamente classificadas como fraude.
*   FN (False Negative): Fraudes não detectadas (modelo classificou como legítimas).

Nesse caso:

*   **Precisão**: mede a proporção de transações que foram corretamente identificadas como fraudulentas entre todas as transações que o modelo previu como fraudulentas. Essa é uma métrica importante, mas transações legítimas sendo classificadas como fraude (ou seja, que são bloqueadas sem necessidade) não são tão problemáticas quando transações fraudulentas que não foram bloqueadas (nesse caso o banco perde dinheiro).


*   **Recall**: mede a proporção de transações fraudulentas reais que foram identificadas corretamente pelo modelo. O recall é mais crítico para a detecção de fraudes, pois queremos que o modelo detecte o máximo possível de fraudes, mesmo que isso signifique cometer alguns falsos positivo. Evitar que fraudes reais passem despercebidas é fundamental para reduzir perdas financeiras

*   **F1-Score**: equilíbrio entre a precisão e recall. Ele é útil quando há um desbalanceamento significativo entre as classes, que é o nosso caso. O F1-score pode ser uma boa métrica, pois fraudes são eventos raros (desbalanceamento de classes).

Então, vamos priorizar o Recall e o F1-Score para esse problema, ou seja, vamos utilizar Regressão Logística sem SelectKBest para esse problema. Logo, as features utilizadas serão retiradas da seguinte lista:

    ['TX_AMOUNT' 'HOUR' 'DAY_OF_WEEK' 'MONTH']


## 4 | Modelagem e Inferência

Testamos alguns modelos fora de uma Pipeline e nessa etapa notei que como o tempo de cada execução estava sendo de mais de uma hora, eu precisaria fazer uma escolha mais certeira, dados que não conseguiria testar todos os modelos e todos os estados (original, padronizado e normalizado) num tempo hábil. Tivemos que limitar o número de features e trees para conseguir um resultado satisfatório e em tempo funcional. Dadas as devidas limitações, montamos uma pipeline considerando: 

*   KNN original e padronizado
*   CART original e padronizado
*   Regressão Logística original e padronizada
*   Random Forest original e padronizada
*   Extra Trees original e padronizada
*   ADA Boost original e padronizada
*   Gradient Boost original e padronizada

E avaliamos cada modelo considerando F1-score e Recall. Além disso, utilizadmos joblib para guardar os processos. Depois de dessa análise e de visualizar os boxplots de cada modelo, concluímos que alguns os modelos são semelhantes em termos de performance das duas métricas que consideramos importantes. Temos alguns pontos importante ao analisar os modelos e tomar uma decisão:

*   `KNN-padr`: esse é um modelo que ajuda na interpretabilidade do problema, tem um f1-score e recall bons, porém como nosso dataset é muito grande o treinamento poderia demorar muito tempo (e pensando num modelo mais adaptativo a concept drifts, por exemplo, seria difícil o processo de treinamento de tempos em tempos, que é o que a gente precisaria nesse caso), esse não é o melhor modelo. Além disso, a possibilidade de overfitting é algo que queremos desviar, principalmente com um dataset dessa magnetude. O modelo é sensível à mudanças de escala, então os dados teriam que ser padronizados caso a escolha fosse essa.

*   `RF-padr`: esse modelo é mais eficiente e escalável, considerando o volume de dados. Além disso, ele lida bem que as 4 features que gostaríamos de utilizar. Ele captura interações não lineares e lida bem com dados com ruído ou desbalanceamento. Portanto usaremos esse ensemble (até para evitar overfitting, já que a combinação de vários modelos ajuda a reduzir o risco de overfitting).

## 5 | Modelagem e Inferência

Como justificado anteriormente, optamos pelo ensemble **Random Forest**, que é uma técnica de Bagging (que combina vários modelos de aprendizagem, em geral árvores de decisão).  

O funcionamento do **Random Forest** é o seguinte: ele cria múltiplas subamostras do conjunto de dados original (no caso, de treinamento) com reposição. Ou seja, pra cada árvore construída ele seleciona aleatoriamente exemplos do conjunto de treino. Depois de construir várias árvores, a previsão final do Random Forest é feita por votação A principal vantagem do Random Forest é que, devido ao bagging, ele consegue reduzir o overfitting. Isso acontece porque ele usa amostras diferentes e subconjuntos diferentes de features, o que reduz a chance de uma única árvore capturar ruídos ou padrões irrelevantes dos dados.

Dado que definimos a escolha do modelo, vamos entender como otimizar os parâmetros para ter o melhor resultado possível.

No caso de **Random Forest (RF)**, os principais parâmetros que podem ser ajustados são:

*   `n_estimators:`  número de árvores. Aumentar o número de árvores tende a melhorar o modelo, porém aumenta também o tempo de treinamento. Vamos utilizar alguns valores e entender como afeta a parformance.

*   `max_depth:`  profundidade máxima das árvores. Árvores muito profundas tem uma chance maior de se adaptar demais ao conjunto de treinamento. Porém árvores muito rasas podem perdem padrões importantes.

*   `min_samples_split:`  número mínimo de amostrar para dividir um nó.

*   `max_features:`  número máximo de features a serem consideradas em cada divisão de nó.

*   `bootstrap:`  define se será usada amostragem com reposição para criar as árvores.

Vamos testar algumas configurações usando **RandomizedSearchCV** em vez de **GridSearchCV**, pois nosso conjunto de dados é muito grande. Além disso, vamos optar por diminuir ainda mais o conjunto de treino (considerando o tempo que a primeira parte de modelagem levou para analisar os dados de treino atuais).

Pela avaliação acima, das 30 combinações testadas considerando a métrica **f1-score** os melhores parâmetros para **Random Forest** padronizada são:

*   rf__n_estimators': 100
*   rf__min_samples_split': 2
*   rf__min_samples_leaf': 5
*   rf__max_features': log2
*   rf__max_depth': 20
*   rf__bootstrap': True

## 6 | Treinamento do Modelo

O modelo foi treinado considerando o conjunto de treino e foi "testado" no conjunto de validação, obtendo os resultados:

*   F1-score de 0,9818: O F1-score é a média harmônica entre precisão e recall. Esse valor sugere que o modelo está sendo eficaz em prever corretamente tanto as classes positivas quanto negativas, sem gerar muitos falsos positivos ou falsos negativos.

*   Recall de 0,9642: O recall mede a capacidade do modelo de identificar corretamente todas as instâncias positivas (ou seja, identificar fraude). Nesse caso, um recall de 96,42% indica que o modelo está conseguindo capturar a maior parte dos exemplos positivos (as instâncias de interesse).

Com o resultado positivo do modelo já treinado, agora vamos fazer um **re-treino** com o conjunto de treino + conjunto de validação e entender os resultados agora no conjunto de teste:

*   f1-score: 0.9823615663274108
*   Recall: 0.9653345764292071

Através da matriz de confusão, notamos que:
  
*Falsos positivos* são inexistentes, o que é ótimo, pois isso significa que não há transações legítimas sendo classificadas erroneamente como fraudulentas, o que poderia significar um mal estar para o cliente.

*Falsos negativos* são relativamente baixos, mas ainda assim o modelo poderia melhorar um pouco nesse aspecto para capturar mais fraudes. Dito isso, num ambiente com uma capacidade computacional maior poderíamos avaliar mais hiperparâmetros e modelos específicos para melhorar o desempenho nesse quesito.

## 7 | Preparação do modelo para produção

Por meio do conjunto de teste, verificamos que alcançamos f1-score de 98,23% e recall de 96,53%, em dados não vistos. Valores semelhantes à essas métricas no conjunto de teste são esperados quando esse modelo estiver executando em produção e fazendo predições para novos dados.

Vamos agora preparar o modelo para utilização em produção. Para isso, vamos treiná-lo com todo o dataset, e não apenas o conjunto de treino.

Nesse caso, vamos usar o **modelo tradicional de treinamento** (sem validação cruzada) pois em produção o modelo treinado através de validação cruzada não pode ser usado diretamente em produção, pois ele é validado com subconjuntos dos dados. Além do alto custo computacional e de tempo (para um conjunto de dados grande de dados).

## 8 | Conclusão, Desafios e Próximos passos

Além das hipóteses propostas inicialmente, tivemos vários insights ao longo do caminho. A parte de limpeza e tratamento de dados me fez chorar lágrimas de sangue, mas foi muito bom, proque aprendi e entendi coisas que até então eram bem teóricas. Mas vamos às hipóteses:


A maior lição tirada desse projeto foi entender o porquê dos datasets reais serem tão mais complicados de treinar. Nesse segundo projeto escolhi um dataset com muitas instâncias para tentar entender as dificuldades de treinamento nesse caso.

O dataset estava praticamente limpo e preparado para treinamento, precisamos apenas fazer alguns ajustes e desmontar a coluna de time para que fosse algo possível de ser processado. Nosso objetivo era criar um modelo que classificasse transações fraudulentas baseado em algumas informações da transações. Utilizamos o processo de **Feature Selection** para tentar identificar quais as variáveis ajudariam a explicar melhor a variável target. Utilizando um pouco de análise exploratória de dados tivemos uma ideia de quais seriam os melhores atributos, mas a etapa de  Feature Selection nos ajudou a entender um pouco mais o relacionamento entre as variáveis.

Inicialmente testamos alguns modelos fora de uma pipeline e considerando apenas os dados originais para ter ideia do tempo de treinamento, já que foi a primeira vez que trabalhava com um conjunto tão grande de instâncias. Após excluir alguns podemos por conta do tempo de treinamento, criamos uma pipeline com modelos que faziam sentido para o nosso problema e depois de testá-los concluímos que **Random Forest** seria ideal considerando o tempo de treinamento e as métricas escolhidas: f1-score e recall, visto que tínhamos classes muito desbalanceadas.

Depois que escolhemos o modelo, otimizamos os hiperparâmetros através de uma varredura apeatória de 30 combinações de parâmetros pré-escolhidos:

*   rf__n_estimators: 100
*   rf__min_samples_split: 2
*   rf__min_samples_leaf: 5
*   rf__max_features: log2
*   rf__max_depth: 20
*   rf__bootstrap: True

Com os parâmetros e modelo ideais definidos, usamos o conjunto de treino para treiná-lo e o conjunto de validação para ter uma ideia inicial da sua perfornance. Depois disso, juntamos validação e treino para um re-treino para que pudéssemos analisar a parformance no conjunto de teste. Por último, para que fosse aplicado em produção, treinamos o modelo definitivo com todo o conjunto (treino, teste e validação).

O maior desafio foi o tempo de procura do modelo através da pipeline. Tive que diminuir um pouco o conjunto de treino utilizado e excluir algumas possibilidades pois o processo estava durando mais de três horas (ou seja, a cada alteração eu tinha que esperar 3 horas para verificar os resultados).

Ao avaliar o modelo definitivo através da matriz de confusão concluímos que:

Falsos positivos são inexistentes, o que é ótimo, pois isso significa que não há transações legítimas sendo classificadas erroneamente como fraudulentas, o que poderia significar um mal estar para o cliente.

Falsos negativos são relativamente baixos, mas ainda assim o modelo poderia melhorar um pouco nesse aspecto para capturar mais fraudes. Dito isso, num ambiente com uma capacidade computacional maior poderíamos avaliar mais hiperparâmetros e modelos específicos para melhorar o desempenho nesse quesito.

Como próximos passos, a ideia é entender quais requisitos computacionais seriam possíveis de serem melhorados no cenário atual da empresa em questão para que tivéssemos tempo e recursos para procurar outros modelos e hiperparâmetros que pudessem nos ajudar a identificar uma maior quantidade de fraudes.





