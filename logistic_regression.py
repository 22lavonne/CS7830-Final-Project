# Adapted tutorial from: https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score

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

# get an iterator that will randomly split the data based on the groups passed in
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(x, y, groups))

# then get the training and testing data based on the splitter
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

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

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)

precision = precision_score(y_test, y_pred)
print(f"Precision: {round(precision,2)}"
)
recall = recall_score(y_test, y_pred)
print(f"Recall: {round(recall,2)}"
)
f1 = f1_score(y_test, y_pred)
print(f"f1 score: {round(f1,2)}"
)
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

# get feature importances
feature_importances = model.coef_[0]

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
plt.show()