import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# code adapted on tutorials from:
# https://www.geeksforgeeks.org/machine-learning/random-forest-regression-in-python/
# https://www.geeksforgeeks.org/machine-learning/calculate-roc-auc-for-classification-algorithm-such-as-random-forest/ 


df = pd.read_csv("dataset.csv")
# print(df)
# df.info()

# all features:
# file,opt_level,is_obfuscated,label,distinctSecurityFlagsCount,fractionOfFunctionsWithFlags,lowRiskFunctionCount,
# mediumRiskFunctionCount,highRiskFunctionCount,totalFunctionCount,internalFunctionCount,externalFunctionCount,uniqueCommunityCount,
# externalFunctionCallFraction,callsVulnerableCount,hasFileIOAPICount,hasNetworkAPICount,hasFilePathCount,socketApiCount,
# connectApiCount,bindApiCount,listenApiCount,acceptApiCount,sendApiCount,sendtoApiCount,recvApiCount,gethostbynameApiCount,
# inet_addrApiCount,setsockoptApiCount,forkApiCount,killApiCount,signalApiCount,prctlApiCount,fopenApiCount,unlinkApiCount,chmodApiCount,opendirApiCount

# networking apis: hasNetworkAPICount, socketApiCount, connectApiCount, bindApiCount, listenApiCount, acceptApiCount, sendApiCount, sendtoApiCount, recvApiCount, 
# gethostbynameApiCount, inet_addrApiCount, setsockoptApiCount

# features determined by ghidrassist: distinctSecurityFlagsCount, fractionOfFunctionsWithFlags, lowRiskFunctionCount, mediumRiskFunctionCount, highRiskFunctionCount, 
# uniqueCommunityCount, callsVulnerableCount

# standard x containing all features (which still excludes the file name and label (where label is y))
x = df.drop(columns=['file', 'label',])

# x where all networking features are dropped
# x = df.drop(columns=['file', 'label', 'hasNetworkAPICount', 'socketApiCount', 'connectApiCount', 'bindApiCount', 'listenApiCount', 'acceptApiCount', 
#                      'sendApiCount', 'sendtoApiCount', 'recvApiCount', 'gethostbynameApiCount', 'inet_addrApiCount', 'setsockoptApiCount'])


# x where all features analyzed/determined by ghidrassist are dropped
# x = df.drop(columns=['file', 'label', 'distinctSecurityFlagsCount', 'fractionOfFunctionsWithFlags', 'lowRiskFunctionCount', 'mediumRiskFunctionCount', 
#                      'highRiskFunctionCount', 'uniqueCommunityCount', 'callsVulnerableCount'])

# x excluding both networking and ghidrassist features:
# x = df.drop(columns=['file', 'label', 'hasNetworkAPICount', 'socketApiCount', 'connectApiCount', 'bindApiCount', 'listenApiCount', 'acceptApiCount', 
#             'sendApiCount', 'sendtoApiCount', 'recvApiCount', 'gethostbynameApiCount', 'inet_addrApiCount', 'setsockoptApiCount', 'distinctSecurityFlagsCount', 'fractionOfFunctionsWithFlags', 'lowRiskFunctionCount', 'mediumRiskFunctionCount', 
#             'highRiskFunctionCount', 'uniqueCommunityCount', 'callsVulnerableCount', 'totalFunctionCount', 'internalFunctionCount', 'hasFilePathCount'])


# y is for the label of malware vs non-malware, where 1 is malware and 0 is benign
y = df.iloc[:,3].values

# define a new parameter for the program name, which is what the splitter will use to 
df["program"] = df['file'].str.split('__').str[9]
groups = df["program"]

# stratified group k fold: keeps each program entirely within one fold, while balancing classes across folds
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

# initialize these to be 0's of length y elements, will be used for stratified k fold
y_prob = np.zeros(len(y))
y_pred = np.zeros(len(y), dtype=int)

for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups)):
    # random forest classifier (no feature scaling needed for tree based models)
    classifier = RandomForestClassifier(n_estimators=100, random_state=1)
    classifier.fit(x.iloc[train_idx], y[train_idx])

    # get the prediction and predicted probability for the current index
    y_prob[test_idx] = classifier.predict_proba(x.iloc[test_idx])[:, 1]
    y_pred[test_idx] = classifier.predict(x.iloc[test_idx])

    # per-fold metrics (AUC only defined if the test fold has both classes)
    y_fold = y[test_idx]
    # get auc of current fold
    fold_auc = roc_auc_score(y_fold, y_prob[test_idx]) if len(np.unique(y_fold)) == 2 else float('nan')
    # print intermediate values for how many malware and benign in the current fold, as well as accuracy, f1, and AUC
    print(f"Fold {fold}: {int(y_fold.sum())} malware / {int((1 - y_fold).sum())} benign in test,",
            "accuracy:", accuracy_score(y_fold, y_pred[test_idx]),
            "F1:", f1_score(y_fold, y_pred[test_idx], zero_division=0),
            "AUC:", fold_auc)

# the metrics below are pooled over all out-of-fold predictions, so y replaces y_test

# confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
# plotting confusion matrix
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# fit one final classifier on all the data to get the feature importances
classifier = RandomForestClassifier(n_estimators=100, random_state=1)
classifier.fit(x, y)

# get the feature importances from the random forest classifier
feature_importances = classifier.feature_importances_

# then put it into a data frame so it can be sorted descending
feature_importances_df = pd.DataFrame({
    'feature': x.columns,
    'coefficient': feature_importances,
}).sort_values(by='coefficient', ascending=False)


# showing the importance of features (only positive for random forest)
plt.barh(feature_importances_df['feature'], feature_importances_df['coefficient'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()

# metrics used: confusion matrix, f1 score, accuracy, precision, recall, auc_roc curve, average precision.

# ROC curve
fpr, tpr, thresholds = roc_curve(y, y_prob, pos_label=1)
roc_auc = roc_auc_score(y, y_prob)
# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# roc curve for tpr = fpr 
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
# plt.savefig("test")

# other metrics
f1 = f1_score(y, y_pred, zero_division=0)
avg_prec = average_precision_score(y, y_prob)
avg_prec_ben = average_precision_score(1- y, 1 - y_prob)
avg_prec_total = (avg_prec + avg_prec_ben) / 2

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUC ROC curve value:", roc_auc)
print("Mean average precision:", avg_prec_total)