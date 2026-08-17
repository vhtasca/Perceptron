# Perceptron

Implementação de um Perceptron simples com função de ativação **relé bipolar**, desenvolvida para a disciplina de Inteligência Artificial.

## Sobre

O Perceptron é o modelo mais simples de rede neural artificial, capaz de aprender a separar dados linearmente separáveis em duas classes através de um processo de treinamento supervisionado baseado no ajuste iterativo de pesos.

Neste projeto, a ativação utilizada é a **relé bipolar**, que classifica a saída do neurônio em dois valores possíveis:

```
f(soma) = 1  se soma >= 0
f(soma) = -1 se soma < 0
```

Por isso, as saídas esperadas do conjunto de dados também são representadas como `1` ou `-1`, em vez de `1` e `0`.

## Como funciona

O treinamento segue a regra de aprendizado do Perceptron:

1. Os pesos (`p1`, `p2`) e o bias (`pb`) são inicializados aleatoriamente.
2. Para cada amostra de entrada, é calculada a soma ponderada:
   ```
   soma = (x1 * p1) + (x2 * p2) + (1 * pb)
   ```
3. A soma passa pela função de ativação relé bipolar, gerando a saída obtida (`1` ou `-1`).
4. O erro é calculado como a diferença entre a saída desejada e a saída obtida.
5. Os pesos são ajustados proporcionalmente ao erro e à taxa de aprendizado:
   ```
   peso = peso + taxa_aprendizado * erro * entrada
   ```
6. O processo se repete a cada amostra, por várias épocas, até que a acurácia atinja 100% (ou até o número máximo de épocas ser alcançado).

Ao final do treinamento, os pesos resultantes são testados novamente contra todas as amostras — dessa vez sem nenhuma atualização de peso — para confirmar de forma independente que a acurácia reportada durante o treino realmente se sustenta com os pesos finais.

## Estrutura

- `Perceptron.predicao(entradas, saidas, taxa_aprendizado, epocas)`: treina o Perceptron e retorna os pesos finais (`p1`, `p2`, `pb`).
- `Perceptron.testar(pesos, entradas, saidas)`: avalia os pesos treinados contra um conjunto de entradas e saídas, exibindo o resultado por amostra e a acurácia final.

## Como executar

```bash
python main.py
```

O programa exibe, para cada época, os detalhes de cada amostra processada (entrada, saída desejada, saída obtida, erro e pesos atualizados), a acurácia da época, e ao final, os pesos resultantes do treinamento e o resultado do teste de verificação.

## Exemplo de dados

```python
entradas = [[0.5, 0.7], [0.8, 1], [-1, 0.2], [1, 0.4]]
saidas = [-1, -1, 1, 1]
```
