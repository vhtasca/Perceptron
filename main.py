import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self):
        pass

    def carregar_amostras(self, caminho_arquivo):
        df = pd.read_csv(caminho_arquivo)

        colunas_entrada = [coluna for coluna in df.columns if coluna != "categoria"]
        entradas = df[colunas_entrada].values.tolist()
        saidas = df["categoria"].apply(lambda c: 1 if c == "A" else -1).tolist()

        return entradas, saidas, colunas_entrada

    def treinamento(self, entradas, saidas, taxa_aprendizado, epocas, bias):
        self.entradas = entradas
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.bias = bias
        melhor_acuracia = 0
        ultima_acuracia = 0 
        p = []

        for _ in range(len(entradas[0])):
            p.append(np.random.uniform(-1, 1))
        pb = np.random.uniform(-1, 1)

        for i in range(epocas):
            acertos = 0  
            for j in range(len(entradas)):
                soma = 0
                for k in range(len(entradas[j])):
                    soma += (p[k]*entradas[j][k])
                soma += bias*pb

                resultado = 1 if soma >= 0 else -1
                erro = saidas[j] - resultado

                if erro == 0:
                    acertos += 1

                for l in range(len(p)):
                    p[l] = p[l] + taxa_aprendizado*erro*entradas[j][l]
                pb = pb + taxa_aprendizado*erro*bias

                print(f"Época: {i}\n"
                    + f"Amostra: {entradas[j]}\n"
                    + f"Saída Desejada: {saidas[j]}\n"
                    + f"Saída Obtida: {resultado}\n"
                    + f"Erro: {erro}\n"
                    + f"Pesos: {p}\nPeso BIAS: {pb:.4f}\n")             

            acuracia = acertos/len(entradas)
            ultima_acuracia = acuracia
            if acuracia > melhor_acuracia:
                melhor_acuracia = acuracia
                melhor_peso = p.copy()
                melhor_bias = pb
            print(f"Acurácia: {acuracia:.2%}")
            print(f"Acertos: {acertos}\n")

            if acuracia == 1:
                break

        if acuracia == 1:
            print("========FIM DO TREINAMENTO========")
            print(f"100% de acurácia atingida na época {i}")
            print(f"Melhores Pesos: {melhor_peso}")
            print(f"Melhor BIAS: {melhor_bias:.4f}")
            print(f"Total de Épocas: {epocas}\n")

        else: 
            print("========FIM DO TREINAMENTO========")
            print(f"Quantidade de Épocas: {epocas}")
            print(f"Melhor acurácia: {melhor_acuracia:.2%}")
            print(f"Última acurácia: {ultima_acuracia:.2%}")    
            print(f"Melhores Pesos: {melhor_peso}")
            print(f"Melhor BIAS: {melhor_bias:.4f}\n")

        return melhor_peso, melhor_bias

    def nova_amostra(self, colunas_entrada):
        print("========NOVA AMOSTRA========")
        entrada = []
        for colunas in colunas_entrada:
            valor = float(input(f"Digite o valor de {colunas}: "))
            entrada.append(valor)
        return entrada
    
    def testar(self, pesos, pb, bias, amostra):
        print("\n========FIM DO TESTE========")
        soma = 0
        for k in range(len(amostra)):
            soma += (pesos[k]*amostra[k])
        soma += bias*pb

        resultado = 1 if soma >= 0 else -1

        print(f"Amostra: {amostra}\n"
            + f"Saída Obtida: {resultado}\n"
            + f"Categoria: {'A' if resultado==1 else 'B' if resultado==-1 else 'ERRO'}\n")
        
        return resultado

percp = Perceptron()

bias = float(input("Digite o valor do BIAS: "))
taxa_aprendizado = (float(input("Digite a taxa de aprendizagem: ")))
epocas = (int(input("Digite o número de épocas: ")))

entradas, saidas, colunas_entrada = percp.carregar_amostras("amostras_1200_lenta.csv")
pesos, pb = percp.treinamento(entradas, saidas, taxa_aprendizado, epocas, bias)

amostra = percp.nova_amostra(colunas_entrada)

percp.testar(pesos, pb, bias, amostra)