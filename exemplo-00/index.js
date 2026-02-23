import tf, { train } from '@tensorflow/tfjs-node';


function gerarDados(quantidade) {
    const xs = [];
    const ys = [];

    for (let i = 0; i < quantidade; i++) {
        // Idade aleatória entre 18 e 60 (normalizada)
        const idadeRaw = Math.floor(Math.random() * (60 - 18 + 1)) + 18;
        const idadeNorm = (idadeRaw - 18) / (60 - 18);

        // Cores (One-hot: azul, vermelho, verde)
        const cores = [0, 0, 0];
        const corIdx = Math.floor(Math.random() * 3);
        cores[corIdx] = 1;

        // Cidades (One-hot: SP, Rio, Curitiba)
        const cidades = [0, 0, 0];
        const cidIdx = Math.floor(Math.random() * 3);
        cidades[cidIdx] = 1;

        // Unindo entradas [idade, azul, vermelho, verde, SP, Rio, Curitiba]
        xs.push([idadeNorm, ...cores, ...cidades]);

        // Lógica de Negócio para a IA aprender:
        // Se for jovem (<30) e de SP, ou gostar de Azul -> Premium [1,0,0]
        // Se for meia idade (30-45) -> Medium [0,1,0]
        // Se for mais velho (>45) -> Basic [0,0,1]
        if (idadeRaw < 30 && (cidIdx === 0 || corIdx === 0)) {
            ys.push([1, 0, 0]);
        } else if (idadeRaw <= 45) {
            ys.push([0, 1, 0]);
        } else {
            ys.push([0, 0, 1]);
        }
    }
    return { 
        xs: tf.tensor2d(xs), 
        ys: tf.tensor2d(ys) 
    };
}

async function trainModel(xs, ys) {
    // Criamos um modelo sequencial simples
    const model = tf.sequential();

    // primeira camada de rede :
    // entrada de 7 características (idade normalizada, 3 corres e 3 localizações)

    // 80 neuronios foi colocado pq tem pouca base de treino
    //quanto mais neuronios, mais complexa a rede, mas pode levar a overfitting com poucos dados
    // e consequentemente, mais procvessamento ela vai usar

    // a Relu  age como um filtro
    // é como se ela deixasse somente os dados interessantes seguirem viagem na rede
    // se a informação chegou  nesse neuronio é positiva, passa para frente, se for negativa, é descartada

    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' })); // Camada oculta com 10 neurônios

    // saída : tres neuronios, um para cada categoria (premium, medium, basic)

    // activation softmax é usada para classificação multiclasse, ela transforma as saídas em probabilidades que somam 1
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // Camada de saída com 3 neurônios (para as 3 categorias)

    // compilando modelo
    // optimizer 'adam' é um algoritmo de otimização eficiente para ajustar os pesos da rede durante o treinamento e aprende com base nos erros cometidos,
    // melhorando as previsões ao longo do tempo

    // loss 'categoricalCrossentropy' ele compara o que oi modelo "acha" (os scores de cada categoria) 
    // com o que é a resposta correta (as labels one-hot encoded) e calcula o erro, ajudando o modelo a aprender a fazer previsões mais precisas
    // a categoria premius tem a label [1, 0, 0], se o modelo prever [0.8, 0.1, 0.1], a função de perda vai calcular o erro entre essas duas distribuições e 
    // ajustar os pesos da rede para reduzir esse erro nas próximas previsões.

    // metrics 'accuracy' quanto mais distante da previsão do modelo da resposta correta, maior o erro (loss)
    // exemplo classico -> classificação de imagem, recomendação de produtos, detecção de fraudes, etc
    // qualquer coisa que a resposta certa é apenas entre varias possiveis

    model.compile({
        optimizer: 'adam', // Otimizador para ajustar os pesos da rede
        loss: 'categoricalCrossentropy', // Função de perda para classificação multiclasse
        metrics: ['accuracy'] // Métrica para avaliar o desempenho do modelo
    });

    // treinamento do modelo
    // epochs 1000 é o numero de vezes que o modelo vai passar por todo o dataset de treino, quanto mais epochs, mais o modelo tem chance de aprender, mas cuidado com overfitting
    // batchSize 32 é o numero de amostras que o modelo vai processar antes de atualizar os pesos, um batch menor pode levar a um treinamento mais ruidoso, mas pode ajudar a escapar de mínimos locais
    await model.fit(xs, ys, {
        verbose: 0, // Desativa a saída detalhada do treinamento
        epochs: 100, // Número de vezes para treinar o modelo
        shuffle: true, // Embaralha os dados a cada época para melhorar a generalização do modelo
        callbacks: { // Callback para monitorar o progresso do treinamento
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    return model;
}

async function predict(model, pessoatensorNormalizada) {

    // transformar o array de entrada em um tensor 2D, pois o modelo espera uma matriz de entradas
    const inputTensor = tf.tensor2d(pessoatensorNormalizada);

    // faz a previsão usando o modelo treinado
    const prediction = model.predict(inputTensor);

    const predArray = await prediction.array(); // Converte o tensor de previsão para um array JavaScript

    return predArray[0].map((prob, index) => ({ prob, index })); // Retorna a previsão para a primeira (e única) pessoa

    // // obtemos os scores de cada categoria (premium, medium, basic)
    // const scores = prediction.dataSync(); // Extrai os valores do tensor de previsão

    // // encontramos o índice da categoria com a maior pontuação
    // const predictedIndex = scores.indexOf(Math.max(...scores));

    // // mapeamos o índice para o nome da categoria correspondente
    // const predictedCategory = labelsNomes[predictedIndex];

    // console.log(`Previsão para ${pessoa.nome}: ${predictedCategory} (scores: ${scores})`);
}


// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

const dadosTreino = gerarDados(200);

const pessoa = { nome: "Evandson", idade: 28, cor: "verde", localizacao: "Curitiba" }

// normalizando a idade da nova pessoa usando o mesmo padrão do treino
// exemplo de normalização: (idade - idade_min) / (idade_max - idade_min)
// (28 - 25) / (40 - 25) = 0.2
const idadeNormalizada = (pessoa.idade - 25) / (40 - 25); // Normalizando a idade

const pessoatensorNormalizada = [
    [
        idadeNormalizada, // idade normalizada
        0, // cor azul
        0, // cor vermelho
        1, // cor verde (one-hot encoded)
        0, // localização São Paulo
        0, // localização Rio
        1 // localização Curitiba
    ]
]

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)


// Treinamos o modelo usando os tensores de entrada e saída
// quanto mais dados melhor
// assim o algoritmo consegue entender melhor as relações entre as características e as categorias, melhorando a precisão das previsões
const model = await trainModel(dadosTreino.xs, dadosTreino.ys);

const prediction = await predict(model, pessoatensorNormalizada);
const results = prediction
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(`Previsão para ${pessoa.nome}:\n${results}`);

// inputXs.print();
// outputYs.print();