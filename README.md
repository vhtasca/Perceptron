# Perceptron

Implementação de um Perceptron simples com função de ativação **relé bipolar**, desenvolvida para a disciplina de Inteligência Artificial. O projeto lê um conjunto de amostras de um arquivo CSV, treina a rede para classificar as amostras em duas categorias (A ou B) e, ao final, permite testar uma nova entrada informada pelo usuário.

## Sobre

O Perceptron é o modelo mais simples de rede neural artificial, capaz de aprender a separar dados linearmente separáveis em duas classes através de um processo de treinamento supervisionado baseado no ajuste iterativo de pesos.

A ativação utilizada é a **relé bipolar**, que classifica a saída do neurônio em dois valores possíveis:

```
f(soma) = 1  se soma >= 0
f(soma) = -1 se soma < 0
```

As categorias do arquivo de amostras (`A` e `B`) são convertidas para essa representação bipolar (`1` e `-1`) no momento da leitura dos dados.

## Como funciona

### Treinamento

1. O usuário informa, no início da execução, o **valor do bias**, a **taxa de aprendizagem** e o **número de épocas**.
2. As amostras são carregadas de um arquivo CSV (`amostras_1200_lenta.csv`), com uma coluna por atributo de entrada (`x1` a `x5`) e uma coluna `categoria` (`A`/`B`).
3. Os pesos — um por atributo de entrada, mais o peso do bias — são inicializados aleatoriamente entre -1 e 1.
4. Para cada amostra, é calculada a soma ponderada:
   ```
   soma = (x1*p1) + (x2*p2) + ... + (x5*p5) + (bias*pb)
   ```
5. A soma passa pela função de ativação relé bipolar, gerando a saída obtida (`1` ou `-1`).
6. O erro é calculado como a diferença entre a saída desejada e a saída obtida, e os pesos são ajustados proporcionalmente ao erro e à taxa de aprendizagem:
   ```
   peso = peso + taxa_aprendizado * erro * entrada
   ```
7. O processo se repete a cada amostra, por várias épocas. O treinamento é interrompido antecipadamente se a acurácia atingir 100% em alguma época.
8. Ao final, são exibidos os pesos, o bias, o número de épocas executadas e a acurácia obtida (melhor e última, caso não tenha atingido 100%).

### Teste de uma nova amostra

Após o treinamento, o programa solicita ao usuário os valores de entrada de uma nova amostra (uma por atributo) e classifica essa amostra usando os pesos treinados, exibindo se ela pertence à categoria A ou B.

## Estrutura

- `Perceptron.carregar_amostras(caminho_arquivo)`: lê o arquivo CSV com pandas, separando os atributos de entrada da coluna de categoria, e converte as categorias (`A`/`B`) para saída bipolar (`1`/`-1`).
- `Perceptron.treinamento(entradas, saidas, taxa_aprendizado, epocas, bias)`: treina o Perceptron e retorna os melhores pesos e o melhor peso de bias encontrados durante o treinamento.
- `Perceptron.nova_amostra(colunas_entrada)`: solicita ao usuário os valores de uma nova amostra a ser classificada.
- `Perceptron.testar(pesos, pb, bias, amostra)`: classifica uma amostra com os pesos treinados e exibe se ela é categoria A ou B.

## Como executar

```bash
python main.py
```

O programa pede o bias, a taxa de aprendizagem e o número de épocas, treina o Perceptron com as amostras do arquivo CSV (exibindo o progresso por amostra e por época), e em seguida solicita os valores de uma nova amostra para classificá-la como categoria A ou B.

## Arquivo de amostras

O arquivo `amostras_1200_lenta.csv` contém 1200 amostras com 5 atributos de entrada (`x1` a `x5`) e uma coluna `categoria` com os valores `A` ou `B`:

```
x1,x2,x3,x4,x5,categoria
-6.7391,-3.2408,3.5545,2.3307,9.0987,A
-1.7720,8.7928,8.5350,4.3114,-9.6108,A
5.9480,8.0151,0.4773,-6.0567,-6.3841,B
...
```
