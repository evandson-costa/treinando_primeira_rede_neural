# 🧠 Classificação de Categorias com TensorFlow.js

Este projeto demonstra a criação, treinamento e execução de uma **Rede Neural Artificial** simples utilizando Node.js e a biblioteca TensorFlow.js. O objetivo é classificar usuários em três categorias (**Premium**, **Medium** ou **Basic**) com base em características como idade, cor favorita e localização.

---

## 🚀 O que este modelo faz?

O modelo recebe um vetor de dados normalizados e processa essas informações através de camadas densas para prever a probabilidade de um perfil pertencer a uma das categorias pré-definidas.

### Estrutura da Rede:
1.  **Camada de Entrada (Input):** Recebe 7 características (Idade normalizada + One-hot encoding de cores e cidades).
2.  **Camada Oculta (Hidden Layer):** * **80 Neurônios**: Responsáveis por encontrar padrões nos dados.
    * **Ativação ReLU**: Funciona como um filtro, deixando passar apenas informações positivas e relevantes.
3.  **Camada de Saída (Output):**
    * **3 Neurônios**: Representam as categorias (Premium, Medium, Basic).
    * **Ativação Softmax**: Transforma os resultados em probabilidades que somam 100%.

---

## 🛠️ Tecnologias Utilizadas

* **Node.js**: Ambiente de execução.
* **TensorFlow.js (@tensorflow/tfjs-node)**: Biblioteca de Machine Learning.
* **Adam Optimizer**: Algoritmo que ajusta os pesos da rede para reduzir o erro.
* **Categorical Crossentropy**: Função de perda ideal para problemas de classificação multiclasse.

---

## 📝 Conceitos Chave Aplicados

> **One-Hot Encoding**: Técnica para transformar dados categóricos (como nomes de cidades) em números que a rede neural consegue processar. Ex: `São Paulo` vira `[1, 0, 0]`.

> **Epochs (Épocas)**: O modelo revisa o conjunto de dados 100 vezes para aprender com os erros e ajustar seus pesos internos.

> **Normalização**: Os dados de idade são convertidos para uma escala entre 0 e 1, facilitando a convergência matemática do treinamento.

---

## 🏁 Como Executar

1. Certifique-se de ter o Node.js instalado.
2. Instale as dependências:
   ```bash
   npm install
