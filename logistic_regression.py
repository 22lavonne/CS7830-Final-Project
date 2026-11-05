# Adapted tutorial from: https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score

df = pd.read_csv("dataset.csv")

x = df.drop(columns=['file', 'label',])
# y is for the label of malware vs non-malware, where 1 is malware and 0 is benign
y = df.iloc[:,3].values

# training and testing split
x_train, x_test, y_train, y_test = train_test_split(
    # same test size and random seed as random forest
    x, y, test_size=0.2, random_state=42
)

# feature scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("y_pred:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {round(accuracy,2)}")

cm = confusion_matrix(y_test, y_pred)
# cm
print("cm:", cm)

precision = precision_score(y_test, y_pred)
f"Precision: {round(precision,2)}"

recall = recall_score(y_test, y_pred)
f"Recall: {round(recall,2)}"

f1 = f1_score(y_test, y_pred)
f"f1 score: {round(f1,2)}"

y_prob = model.predict_proba(x_test)[:, 1]
print("y_prob", y_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
print("fpr:", fpr, "tpr:", tpr, "thresholds:", thresholds)

roc_auc = roc_auc_score(y_test, y_prob)
print("roc_auc:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--") 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()