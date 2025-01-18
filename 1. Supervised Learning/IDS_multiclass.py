"""
@author: Grisolia Giuseppe

Modulo che esegue preprocessing dati presenti nel dataset combinato 'combined_dataset.csv',
ottenuto da concat.py, addestra modello Random forest e ne valuta i risultati.
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, roc_auc_score,
                             classification_report, confusion_matrix, accuracy_score, f1_score)
from sklearn.preprocessing import label_binarize

# Carica il dataset combinato
combined_dataset = pd.read_csv('../1. Supervised Learning/combined_dataset.csv')
dataset = combined_dataset.dropna()

# esplorazione del dataset
print(dataset.info())

# Conversione del campo CAN ID da esadecimale a decimale
dataset.loc[:, 'CAN_ID'] = dataset['CAN_ID'].apply(lambda x: int(x, 16))

# Conversione dei campi da Data[0] a Data[7] da esadecimale a decimale
for i in range(8):
    col_name = 'Data[' + str(i) + ']'
    dataset.loc[:, col_name] = dataset[col_name].apply(lambda x: int(x, 16))

# divido i dati in features di input e feature target
y = dataset['Flag']
X = dataset.drop('Flag', axis=1)

# divido il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

# creazione classificatore random forest
clf = RandomForestClassifier(max_depth=int(10), random_state=int(42), n_estimators=20)
clf.fit(X_train, y_train)

# Effettua previsioni sul test set
prediction = clf.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print('\naccuracy_score:', accuracy)

# Stampa del rapporto di classificazione e della matrice di confusione
print('\nClasification report:\n', classification_report(y_test, prediction))
print('\nConfussion matrix:\n', confusion_matrix(y_test, prediction))

# Creazione e visualizzazione della matrice di confusione come heatmap
confusion_matrix = confusion_matrix(y_test, prediction)
class_labels = [str(i) for i in range(5)]  # Genera le etichette "0", "1", "2", "3", "4"
df_cm = pd.DataFrame(confusion_matrix, index=class_labels, columns=class_labels)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
pyplot.show()

# Calcolo delle probabilità e dell'AUC per la curva ROC
probs = clf.predict_proba(X_test)
# Calcola l'AUC per ciascuna classe
auc = roc_auc_score(y_test, probs, multi_class='ovr')
print('AUC: %.3f' % auc)

# Disegna le curve ROC per ogni classe
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
for i in range(probs.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs[:, i])
    plt.plot(fpr, tpr, marker='.', label=f'Class {i} vs Rest')

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Calcolo dell'average precision e visualizzazione della curva precision-recall per ogni classe
for i in range(probs.shape[1]):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], probs[:, i])
    average_precision = average_precision_score(y_test_binarized[:, i], probs[:, i])

    plt.step(recall, precision, where='post', label=f'Class {i} (AP={average_precision:.2f})')
    plt.fill_between(recall, precision, step='post', alpha=0.2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Calcolo f1 score
f1 = f1_score(y_test, prediction, average='weighted')
print('\nf1 score: ', f1)

'''
# Valutazione del modello attraverso cross-validation (di 3)
cv_scores = cross_val_score(clf, X, y, cv=3)

# Stampa delle statistiche ottenute dalla cross-validation
print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
print('\ncv_score variance:{}'.format(np.var(cv_scores)))
print('\ncv_score dev standard:{}'.format(np.std(cv_scores)))
print('\n')

# Creazione di un grafico a barre per visualizzare varianza e deviazione standard dei cv_scores
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()
'''
