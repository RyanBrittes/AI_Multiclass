from compairClasses import CompairClasses

predict = CompairClasses()

predicted_values = predict.compair_one_vs_all()

print(f"Pesos: {predicted_values[2]}")