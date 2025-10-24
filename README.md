<h1 align="center">🌸 Regressão Logística com o Dataset Iris 🌸</h1>

<p align="center">
  <img src="https://78.media.tumblr.com/5dace9e7bd0e72c2f3c90cf652321f34/tumblr_p49mbeKyO81uvnto5o1_1280.gif" width="400px" alt="Iris GIF">
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python Badge">
  <img src="https://img.shields.io/badge/scikit--learn-Modelagem%20ML-orange?logo=scikit-learn" alt="Scikit-learn Badge">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License Badge">
  <img src="https://img.shields.io/badge/Status-Ativo-brightgreen" alt="Status Badge">
</p>


---

## 📘 Sobre o Projeto

Este projeto demonstra a aplicação de **Regressão Logística Multinomial** utilizando o famoso **conjunto de dados Iris**, uma base clássica para tarefas de classificação.  
Além de treinar o modelo, o código realiza **padronização dos dados**, **avaliação do desempenho**, **visualização da matriz de confusão** e análise das **probabilidades previstas** para uma amostra de teste.

---

##  O Que São *Setosa*, *Versicolor* e *Virginica*?

Esses são os **nomes das três espécies de flores** presentes no conjunto de dados **Iris**, coletado originalmente por **Ronald Fisher (1936)**:

| Espécie       | Características principais |
|----------------|----------------------------|
| **Iris setosa** | Pétalas pequenas e sépalas curtas. É a espécie mais fácil de diferenciar. |
| **Iris versicolor** | Tamanho intermediário entre as outras duas. Pétalas médias. |
| **Iris virginica** | Maior das três, com pétalas e sépalas longas. |

Cada linha do dataset representa **uma flor medida em laboratório**, com quatro atributos numéricos:

1. **sepal length (cm)** — comprimento da sépala  
2. **sepal width (cm)** — largura da sépala  
3. **petal length (cm)** — comprimento da pétala  
4. **petal width (cm)** — largura da pétala  

A variável-alvo (`species`) é uma das três espécies acima.

---

##  O Que o Código Faz Passo a Passo

1. **Importa bibliotecas** – pandas, numpy, seaborn, matplotlib e scikit-learn.  
2. **Carrega o dataset Iris** com `load_iris()` (um dataset embutido no Scikit-learn).  
3. **Separa as variáveis**:
   - `X`: características das flores (as 4 medidas).
   - `y`: rótulos (as espécies).  
4. **Divide os dados** em treino e teste com `train_test_split()` (70% treino, 30% teste).  
5. **Padroniza os dados** com `StandardScaler` (transforma as medidas para uma escala comparável).  
6. **Treina um modelo de Regressão Logística Multiclasse** com:
   ```python
   LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
   ```

Isso significa que o modelo aprende a prever qual das três espécies uma flor pertence, com base nas quatro medidas.

7. **Gera previsões e calcula métricas como precisão, recall e F1-score.**
8. **Cria uma Matriz de Confusão, mostrando onde o modelo acertou ou errou as classificações.**
9. **Exibe os coeficientes e interceptos do modelo (como cada medida influencia a predição).**
10. **Seleciona uma amostra de teste e mostra as probabilidades de pertencer a cada espécie.**

---


##  Tecnologias Utilizadas

- 🐍 **Python**
- 📊 **Pandas** e **NumPy** — Manipulação e análise de dados  
- 🤖 **Scikit-learn** — Criação e avaliação do modelo de regressão logística  
- 🎨 **Seaborn** e **Matplotlib** — Visualização da matriz de confusão  

---

##  Como Executar

1. **Clone o repositório**  
   ```bash
   git clone https://github.com/seu-usuario/projeto-regressao-logistica-iris.git
   cd projeto-regressao-logistica-iris
    ```
   
Instale as dependências

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

```bash
python main.py
```

---

📊 Resultados Gerados

Relatório de classificação com métricas de precisão, recall e F1-score

Matriz de confusão visualizada com Seaborn

Exibição dos coeficientes e interceptos do modelo

Probabilidade prevista para uma amostra de teste

---

💡 Observações

O modelo utiliza padronização (StandardScaler) para melhorar a performance.

O parâmetro multi_class='multinomial' permite que o modelo trate a tarefa como uma classificação multiclasse real.

O nome “Kovalski relatório” é apenas um toque de humor no código 😄

---

🧠 Conceitos Envolvidos

Regressão Logística Multiclasse

Normalização e Escalonamento de Dados

Avaliação de Modelos de Classificação

Visualização de Dados com Seaborn
