import pandas as pd
from sklearn.metrics import f1_score

answer = pd.read_csv('test_answer.csv')['label'].values.tolist()
predict = pd.read_csv('sbm.csv')['label'].values.tolist()

print(f1_score(y_pred=predict, y_true=answer, average='weighted'))