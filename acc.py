import pandas as pd

answer = pd.read_csv('test_answer.csv')['label'].values.tolist()
predict = pd.read_csv('sbm.csv')['label'].values.tolist()
TP = {'rock': 0, 'pop': 0, 'hiphop': 0, 'electronic': 0, 'folk': 0}
TN = {'rock': 0, 'pop': 0, 'hiphop': 0, 'electronic': 0, 'folk': 0}
FP = {'rock': 0, 'pop': 0, 'hiphop': 0, 'electronic': 0, 'folk': 0}
FN = {'rock': 0, 'pop': 0, 'hiphop': 0, 'electronic': 0, 'folk': 0}
labels = ['rock', 'pop', 'hiphop', 'electronic', 'folk']

for i, (a, p) in enumerate(zip(answer, predict)):

     if a == p:
          TP[a] += 1



     tmp = list(filter(lambda x: x != a, labels))
     tmp = list(filter(lambda x: x != p, tmp))
     for j in tmp:
        TN[j] += 1

     if a != p:
         FP[p] += 1
     if a != p:
        FN[a] += 1

Precisionlist = []
Recalllist = []
for i in labels:
    Precisionlist.append(TP[i]/(TP[i] + FP[i]))
    Recalllist.append(TP[i]/(TP[i] + FN[i]))

for i, _ in enumerate(labels):
    print(labels[i] + ':', 2 * Precisionlist[i] * Recalllist[i]/(Precisionlist[i] + Recalllist[i]))