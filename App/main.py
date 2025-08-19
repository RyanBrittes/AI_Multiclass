from compairClasses import CompairClasses
import numpy as np

predict = CompairClasses()

predicted_values = predict.compair_one_vs_all()

print(f"Pesos\nClasse 01: {predicted_values[2][0]}\nClasse 02: {predicted_values[2][1]}")
print(f"Verdadeiro: {predicted_values[1]}\nPrevisões: {predicted_values[0]}")
print(f"Acurácia: {np.mean(predicted_values[0] == predicted_values[1])}")
