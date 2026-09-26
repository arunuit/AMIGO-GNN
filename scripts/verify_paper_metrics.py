import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

cm = np.array([
    [196, 3, 1],
    [4, 191, 5],
    [2, 4, 194]
])
y_true, y_pred = [], []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        y_true.extend([i] * int(cm[i,j]))
        y_pred.extend([j] * int(cm[i,j]))

print("Samples:", len(y_true))
print("Correct:", int(np.trace(cm)))
print("Accuracy:", accuracy_score(y_true,y_pred)*100)
print("Macro precision:", precision_score(y_true,y_pred,average="macro")*100)
print("Macro recall:", recall_score(y_true,y_pred,average="macro")*100)
print("Macro F1:", f1_score(y_true,y_pred,average="macro")*100)
print("Multiclass MCC:", matthews_corrcoef(y_true,y_pred))
print("\nThese values are derived ONLY from the Figure-6 matrix in the manuscript.")
print("If the manuscript reports different headline metrics, regenerate the metrics and figure from the same predictions.")
