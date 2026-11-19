# Adapted from following tutorial: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

df = pd.read_csv("dataset.csv")

# x: all parameters
x = df.drop(columns=['file', 'label',])
# x excluding networking features
# x = df.drop(columns=['file', 'label', 'hasNetworkAPICount', 'socketApiCount', 'connectApiCount', 'bindApiCount', 'listenApiCount', 'acceptApiCount', 
#                      'sendApiCount', 'sendtoApiCount', 'recvApiCount', 'gethostbynameApiCount', 'inet_addrApiCount', 'setsockoptApiCount'])

# x excluding ghidrassist features
# x = df.drop(columns=['file', 'label', 'distinctSecurityFlagsCount', 'fractionOfFunctionsWithFlags', 'lowRiskFunctionCount', 'mediumRiskFunctionCount', 
#                      'highRiskFunctionCount', 'uniqueCommunityCount', 'callsVulnerableCount'])

# x excluding both networking and ghidrassist features: (still 100% accurate)
# x = df.drop(columns=['file', 'label', 'hasNetworkAPICount', 'socketApiCount', 'connectApiCount', 'bindApiCount', 'listenApiCount', 'acceptApiCount', 
#             'sendApiCount', 'sendtoApiCount', 'recvApiCount', 'gethostbynameApiCount', 'inet_addrApiCount', 'setsockoptApiCount', 'distinctSecurityFlagsCount', 'fractionOfFunctionsWithFlags', 'lowRiskFunctionCount', 'mediumRiskFunctionCount', 
#             'highRiskFunctionCount', 'uniqueCommunityCount', 'callsVulnerableCount', 'totalFunctionCount', 'internalFunctionCount', 'hasFilePathCount'])

# y is for the label of malware vs non-malware, where 1 is malware and 0 is benign
y = df.iloc[:,3].values

# getting the different types of programs and identifying the groups of them
df["program"] = df['file'].str.split('__').str[9]
groups = df["program"]

# stratified group k fold: keeps each program entirely within one fold, while balancing classes across folds
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

# initialize these to be 0's of length y elements, will be used for stratified k fold
y_prob = np.zeros(len(y))
y_pred = np.zeros(len(y), dtype=int)

for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups)):
    # scaler is fit on the training portion of each fold only
    # use make pipleine method to do this
    # use feature scaling here for neural networks
    clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=1, max_iter=300))
    clf.fit(x.iloc[train_idx], y[train_idx])

    # get the prediction and predicted probability for the current index
    y_prob[test_idx] = clf.predict_proba(x.iloc[test_idx])[:, 1]
    y_pred[test_idx] = clf.predict(x.iloc[test_idx])

    # per-fold metrics (AUC only defined if the test fold has both classes)
    y_fold = y[test_idx]
    # get auc of current fold
    fold_auc = roc_auc_score(y_fold, y_prob[test_idx]) if len(np.unique(y_fold)) == 2 else float('nan')
    # print intermediate values for how many malware and benign in the current fold, as well as accuracy, f1, and AUC
    print(f"Fold {fold}: {int(y_fold.sum())} malware / {int((1 - y_fold).sum())} benign in test,",
            "accuracy:", accuracy_score(y_fold, y_pred[test_idx]),
            "F1:", f1_score(y_fold, y_pred[test_idx], zero_division=0),
            "AUC:", fold_auc)


conf_matrix = confusion_matrix(y, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# plt.show()

# metrics used: confusion matrix, f1 score, accuracy, precision, recall, auc_roc curve, mean average precision.

fpr, tpr, thresholds = roc_curve(y, y_prob)

# fit one final pipeline on all the data to get the training loss curve
clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=1, max_iter=300))
clf.fit(x, y)

# plotting the training log loss curve
plt.figure()
plt.plot(clf[-1].loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Training log loss")
plt.title("MLP Training Loss Curve")
plt.show()

f1 = f1_score(y, y_pred, zero_division=0)
auc_roc = roc_auc_score(y, y_prob)
avg_prec = average_precision_score(y, y_prob)
avg_prec_ben = average_precision_score(1- y, 1 - y_prob)
avg_prec_total = (avg_prec + avg_prec_ben) / 2
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_roc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--") 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
# plt.show()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUC ROC curve value:", auc_roc)
print("Average precision score:", avg_prec_total)