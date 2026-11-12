# Adapted from following tutorial: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.inspection import permutation_importance

df = pd.read_csv("dataset.csv")

# x: all parameters
x = df.drop(columns=['file', 'label',])
# y is for the label of malware vs non-malware, where 1 is malware and 0 is benign
y = df.iloc[:,3].values

# getting the different types of programs and identifying the groups of them
df["program"] = df['file'].str.split('__').str[9]
groups = df["program"]

# get an iterator that will randomly split the data based on the groups passed in
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(x, y, groups))

# then get the training and testing data based on the splitter
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
y_pred = clf.predict(x_test)


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# metrics used: confusion matrix, f1 score, accuracy, precision, recall, auc_roc curve, average precision.

# calculating the permutation importance 
perm_importance = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# put it into a data frame so it can be sorted (smallest to biggest)
perm_importance_df = pd.DataFrame({
    'feature': x.columns,
    'importance': perm_importance.importances_mean
}).sort_values(by='importance', ascending=False)

# plotting the permutation importances (where the higher the importance, 
# it means that if that value shuffles then the accuracy of the model drops more)
plt.figure(figsize=(10, len(perm_importance_df) * 0.3))
plt.barh(perm_importance_df['feature'], perm_importance_df['importance'], color='teal')
plt.xlabel('Mean Accuracy Drop (Permutation Importance)')
plt.ylabel('Features')
plt.title('Neural Network Permutation Importances')
plt.tight_layout()
plt.show()

f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)
prec_score = average_precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUC ROC curve value:", auc_roc)
print("Average precision score:", prec_score)