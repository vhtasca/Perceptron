import numpy as np

class Perceptron:
    def __init__(self):
        pass

    def predicao(self, entradas, saidas, taxa_aprendizado, epocas):
        self.entradas = entradas
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        melhor_acuracia = 0
        ultima_acuracia = 0 

        p1, p2, pb = np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)

        for i in range(epocas):
            acertos = 0  
            for j in range(len(entradas)):
                x1, x2 = entradas[j]
                soma = ((x1*p1) + (x2*p2) + (1*pb))
                resultado = 1 if soma >= 0 else -1
                erro = saidas[j] - resultado

                if erro == 0:
                    acertos += 1

                p1 = p1 + taxa_aprendizado*erro*x1
                p2 = p2 + taxa_aprendizado*erro*x2
                pb = pb + taxa_aprendizado*erro*1


                print(f"Época: {i}\n"
                    + f"Amostra: {entradas[j]}\n"
                    + f"Saída Desejada: {saidas[j]}\n"
                    + f"Saída Obtida: {resultado}\n"
                    + f"Erro: {erro}\n"
                    + f"Pesos: [{p1:.4f}, {p2:.4f}, {pb:.4f}]\n")             

            acuracia = acertos/len(entradas)
            ultima_acuracia = acuracia
            if acuracia > melhor_acuracia:
                melhor_acuracia = acuracia
                melhor_peso = [p1, p2, pb]
            print(f"Acurácia: {acuracia:.2%}")
            print(f"Acertos: {acertos}\n")

            if acuracia == 1:
                break

        if acuracia == 1:
            print("========FIM DO TREINAMENTO========")
            print(f"100% de acurácia atingida na época {i}")
            print(f"Pesos: [{p1:.4f}, {p2:.4f}, {pb:.4f}]\n")
            
        else: 
            print("========FIM DO TREINAMENTO========")
            print(f"Melhor acurácia: {melhor_acuracia:.2%}")
            print(f"Última acurácia: {ultima_acuracia:.2%}")    
            print(f"Melhores Pesos: [{melhor_peso[0]:.4f}, {melhor_peso[1]:.4f}, {melhor_peso[2]:.4f}]\n")

        return p1, p2, pb

    def testar(self, pesos, entradas, saidas):
        p1, p2, pb = pesos
        acertos = 0
        print("========INÍCIO DOS TESTES========")
        for j in range(len(entradas)):
            x1, x2 = entradas[j]
            soma = ((x1*p1) + (x2*p2) + (1*pb))
            resultado = 1 if soma >= 0 else -1
            correto = resultado == saidas[j]

            if correto:
                acertos +=1
            print(f"Amostra: {entradas[j]}\n"
                + f"Saída Desejada: {saidas[j]}\n"
                + f"Saída Obtida: {resultado}\n"
                + f"{'OK' if correto else 'ERRO'}\n")
               
        print("========FIM DOS TESTES========")            
        acuracia = acertos/len(entradas)
        print(f"Acurácia: {acuracia:.2%}")
        print(f"Pesos: [{p1:.4f}, {p2:.4f}, {pb:.4f}]\n")
        return acuracia

entradas = [[0.5, 0.7], [0.8, 1], [-1, 0.2], [1, 0.4]]
saidas = [-1, -1, 1, 1]

percp = Perceptron()
pesos = percp.predicao(entradas, saidas, 0.01, 10)
percp.testar(pesos, entradas, saidas)
