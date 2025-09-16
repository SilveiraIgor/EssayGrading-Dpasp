# Experiment: accuracy curve for Add MNIST.
import pasp, numpy as np
import pandas as pd

def init():
  "Initializes test labels, and accuracy lists."
  from datasets import load_dataset
  #ds = load_dataset("kamel-usp/aes_enem_dataset", "JBCS2025", cache_dir="/tmp/aes_enem")
  ds1 = load_dataset("igorcs/C1-A")
  ds2 = load_dataset("igorcs/C1-B")
  test1, test2 = ds1['test'], ds2['test']
  labels_syntax_a, labels_mistakes_a = [], []
  labels_syntax_b, labels_mistakes_b = [], []
  for linha in test1:
      labels_syntax_a.append(linha['syntax'])
      labels_mistakes_a.append(linha['mistakes'])
  for linha in test2:
      labels_syntax_b.append(linha['syntax'])
      labels_mistakes_b.append(linha['mistakes'])
  accuracy = []
  accuracy_program = []
  assert len(labels_syntax_a) == len(labels_syntax_b) == len(labels_mistakes_a) == len(labels_mistakes_b)
  return (labels_syntax_a, labels_mistakes_a, labels_syntax_b, labels_mistakes_b), accuracy, accuracy_program

if __name__ == "__main__":
  L, A, A_p = init()
  P = pasp.parse("programas-c1/experimento2.pasp")
  iteracao = 0
  lista_dic = []
  dic = {}

  def step_syntax(self):
    "Step callback function for each iteration of training."
    #print(self.pr())
    global dic
    Y = np.argmax(self.pr(), axis=1)
    #print(Y[:len(Y)//2])
    Y = Y[:len(Y)//2]
    #print("Tamanho do vetor no syntax: ", len(Y))
    dic['y_hat_syntax'] = Y
    lista_dic.append(dic)
    dic = {}
    #A.append(np.sum(Y == L)/len(Y))
    #Y_b = Y[:(h := len(Y)//2)]
    #A_p.append(np.sum(Y_b == L_b)/len(Y_b))

  def step_mistakes(self):
    "Step callback function for each iteration of training."
    #print(self.pr())
    global iteracao, dic
    iteracao += 1
    dic['iteracao'] = iteracao
    Y = np.argmax(self.pr(), axis=1)
    #print(Y[:len(Y)//2])
    Y = Y[:len(Y)//2]
    #print("Tamanho do vetor no mistakes: ", len(Y))
    dic['y_hat_mistakes'] = Y
    # Pass step as the step callback function.
  print("Numero de NA: ", len(P.NA))
  P.NA[0].set_step_callback(step_mistakes)
  P.NA[1].set_step_callback(step_syntax)
  # Run the program to learn parameters.
  P()
  print(lista_dic)
  # A and A_p are accuracies of embedded neural network, and program (sum).
  df = pd.DataFrame(lista_dic)
  df.to_csv("subredes-C1.csv", index=False)
