import pasp
import numpy as np
import pandas as pd
from datasets import load_dataset
def get_dataset():
    ds = load_dataset("kamel-usp/aes_enem_dataset", "JBCS2025", cache_dir="tmp/aes_enem")
    train = ds['test']
    y = []
    for linha in train:
        y.append(linha['grades'][0]//40)
    return np.array(y)

y = get_dataset().flatten().tolist()
print("As 10 primeiras respostas: ", y[:10])
print("Tamanho do y: ", len(y))
accs = []
lista_dic = []
for i in range(1,4):
    dic = {}
    dic['epoch'] = i
    dic['y'] = y
    P = pasp.parse(f"programas-c1/C1-{i}iter.pasp")
    #P = pasp.parse(f"programas-c1/experimento.pasp")
    R = P(quiet=True)
    #print(R[:7])
    y_hat = np.argmax(R, axis=1)
    #print(y_hat[:7])
    #print("Tamanho do y_hat: ", len(y_hat.tolist()))
    if len(y_hat.tolist()) == len(y):
        y_hat = y_hat.flatten().tolist()
        dic['y_hat'] = y_hat
        match = 0
        for idx in range(len(y)):
            if y[idx] == y_hat[idx]:
                match += 1
        acc = 100*match/len(y)
        dic['acc'] = acc
        print(f"acc: {acc}")
        lista_dic.append(dic)
        accs.append(acc)
    #print("Vetor de igualdade: ", y==y_hat)
    #print(np.sum(y == y_hat))
    #print(len(y))
    #acc = np.sum(y == y_hat)/len(y)
    #print(acc)
    #accs.append(acc)
print("Fim")
print("Accs: ", accs)
df = pd.DataFrame(lista_dic)
df.to_csv("respostas-c1.csv", index=False)
