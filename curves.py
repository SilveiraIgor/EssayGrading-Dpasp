# Experiment: accuracy curve for Add MNIST.
import pasp, numpy as np

def init():
  "Initializes test labels, and accuracy lists."
  from datasets import load_dataset
  ds = load_dataset("kamel-usp/aes_enem_dataset", "JBCS2025", cache_dir="/tmp/aes_enem")
  test = ds['test']
  labels = []
  for linha in test:
      labels.append(linha['grades'][0]//40)
  accuracy = []
  accuracy_program = []
  return np.array(labels[:2]), accuracy, accuracy_program

if __name__ == "__main__":
  L, A, A_p = init()
  P = pasp.parse("experimento.pasp")

  def step(self):
    "Step callback function for each iteration of training."
    Y = np.argmax(self.pr(), axis=1)
    print(Y)
    A.append(np.sum(Y == L)/len(Y))
    Y_b = Y[:(h := len(Y)//2)]
    A_p.append(np.sum(Y_b == L_b)/len(Y_b))

  # Pass step as the step callback function.
  P.NA[0].set_step_callback(step)
  # Run the program to learn parameters.
  P()
  # A and A_p are accuracies of embedded neural network, and program (sum).
  A, A_p = np.array(A), np.array(A_p)
  print(A)
  print(A_p)
