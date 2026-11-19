# Partially adapted tutorial from: https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, f1_score, roc_curve, roc_auc_score, average_precision_score

df = pd.read_csv("dataset.csv")

# x including all features
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

# define a new parameter for the program name, which the group k fold will be based on
df["program"] = df['file'].str.split('__').str[9]
groups = df["program"]

for g in groups:
    print(g)

# stratified group k fold: keeps each program entirely within one fold, while balancing classes across folds
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

# initialize these to be 0's of length y elements, will be used for stratified k fold
y_prob = np.zeros(len(y))
y_pred = np.zeros(len(y), dtype=int)

for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups)):
    # scaler is fit on the training portion of each fold only
    # use make pipleine method to do this
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x.iloc[train_idx], y[train_idx])

    # get the prediction and predicted probability for the current index
    y_prob[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
    y_pred[test_idx] = model.predict(x.iloc[test_idx])

    # per-fold metrics (AUC only defined if the test fold has both classes)
    y_fold = y[test_idx]
    # get roc auc of current fold
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
plt.show()

# metrics used: confusion matrix, f1 score, accuracy, precision, recall, auc_roc curve, average precision.

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

avg_prec = average_precision_score(y, y_prob)
avg_prec_ben = average_precision_score(1- y, 1 - y_prob)
avg_prec_total = (avg_prec + avg_prec_ben) / 2

fpr, tpr, thresholds = roc_curve(y, y_prob)
# print("fpr:", fpr, "tpr:", tpr, "thresholds:", thresholds)

auc_roc = roc_auc_score(y, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUC ROC curve value:", auc_roc)
print("Mean precision Average:", avg_prec_total)


plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_roc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--") 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
# plt.show()

# get feature importances (fit one final pipeline on all the data)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(x, y)
feature_importances = model[-1].coef_[0]

# get a data frame of the features importances, sort them descending
feature_importances_df = pd.DataFrame({
    'feature': x.columns,
    'coefficient': feature_importances,
}).sort_values(by='coefficient', ascending=False)

# plotting the feature importances descending (includes both positive and negative values)
plt.barh(feature_importances_df['feature'], feature_importances_df['coefficient'])
plt.xlabel('Feature importance')
plt.ylabel('Features')
plt.title('Feature Importance for Logistic Regression Classification')
plt.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
plt.tight_layout()
# plt.show()