import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# code based on tutorials from:
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

# get the features for x, which include all columns except 'file' (which is the name of the file) 
# and 'label' which is the malware vs non malware indicator for y
# list of parameters removed for testing, but it was all still 100% accurate
# 'file', 'label', 'sendApiCount', 'socketApiCount', 'hasNetworkAPICount', 'fractionOfFunctionsWithFlags', 'recvApiCount', 'distinctSecurityFlagsCount', 'inet_addrApiCount', 'connectApiCount', 'highRiskFunctionCount', 'setsockoptApiCount', 'uniqueCommunityCount', 'internalFunctionCount', 'totalFunctionCount'

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

# get an iterator that will randomly split the data based on the groups passed in
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(x, y, groups))

# then get the training and testing data based on the splitter
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# random forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(x_train, y_train)

# predicted class probabilities
y_pred = classifier.predict(x_test)

# predicted probabilities
y_prob = classifier.predict_proba(x_test)[:, 1]

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
# plotting confusion matrix
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

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
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = roc_auc_score(y_test, y_prob)
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
f1 = f1_score(y_test, y_pred)
prec_score = average_precision_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUC ROC curve value:", roc_auc)
print("Average precision score:", prec_score)