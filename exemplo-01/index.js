import tf from '@tensorflow/tfjs-node';

async function rodarPrevisorChurn() {
    console.log("Iniciando o sistema de previsão...");
    
    // 1. Criando a Arquitetura da Rede
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [3], units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // 2. Dados de Treino (Uso, Suporte, Contrato)
    const xs = tf.tensor2d([
        [0.1, 0.9, 0.05], [0.2, 0.8, 0.1], // Perfis que tendem ao Cancelamento
        [0.9, 0.1, 0.8], [0.8, 0.2, 0.9]   // Perfis que tendem à Fidelidade
    ]);
    const ys = tf.tensor2d([
        [1, 0], [1, 0], // Saída: Churn
        [0, 1], [0, 1]  // Saída: Retido
    ]);

    console.log("Treinando o modelo...");

    // 3. Treinamento
    await model.fit(xs, ys, {
        epochs: 100,
        verbose: 0,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if ((epoch + 1) % 20 === 0) {
                    console.log(`Época ${epoch + 1}: Perda = ${logs.loss.toFixed(4)}`);
                }
            }
        }
    });

    console.log("Treinamento finalizado!");

    // 4. Predição: Vamos testar um cliente com risco de sair
    // [Baixo uso (0.15), Alto suporte (0.85), Cliente novo (0.1)]
    const novoCliente = tf.tensor2d([[0.75, 0.05, 0.1]]);
    const predicao = model.predict(novoCliente);
    const resultado = await predicao.data();

    console.log("\n--- Resultado da Análise de Churn ---");
    console.log(`Probabilidade de Cancelar: ${(resultado[0] * 100).toFixed(2)}%`);
    console.log(`Probabilidade de Ficar: ${(resultado[1] * 100).toFixed(2)}%`);
}

rodarPrevisorChurn();