from compairClasses import CompairClasses
import numpy as np

predict = CompairClasses()

predicted_values = predict.compair_one_vs_all()

print(f"------\nPesos\nClasse 01: {predicted_values[2][0][1:]}\nClasse 02: {predicted_values[2][1][1:]}")
print(f"------\nViés\nClasse 01: {predicted_values[2][0][0]}\nClasse 02: {predicted_values[2][1][0]}")
print(f"------\nAcurácia: {np.mean(predicted_values[0] == predicted_values[1])}")
