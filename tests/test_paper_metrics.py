import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef

def test_figure6_matrix():
    cm=np.array([[196,3,1],[4,191,5],[2,4,194]])
    y,p=[],[]
    for i in range(3):
        for j in range(3):
            y += [i]*int(cm[i,j]); p += [j]*int(cm[i,j])
    assert len(y)==600
    assert np.isclose(accuracy_score(y,p),581/600)
    assert np.isclose(matthews_corrcoef(y,p),0.9525158754)
