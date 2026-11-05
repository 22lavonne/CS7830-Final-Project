import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

# tutorial from: https://www.geeksforgeeks.org/machine-learning/random-forest-regression-in-python/

# warnings.filterwarnings('ignore')

df = pd.read_csv("dataset.csv")
# print(df)
# df.info()

# get the features for x, which include all columns except 'file' (which is the name of the file) 
# and 'label' which is the malware vs non malware indicator for y
# list of parameters removed for testing, but it was all still 100% accurate
# 'file', 'label', 'sendApiCount', 'socketApiCount', 'hasNetworkAPICount', 'fractionOfFunctionsWithFlags', 'recvApiCount', 'distinctSecurityFlagsCount', 'inet_addrApiCount', 'connectApiCount', 'highRiskFunctionCount', 'setsockoptApiCount', 'uniqueCommunityCount', 'internalFunctionCount', 'totalFunctionCount'
x = df.drop(columns=['file', 'label',])
# y is for the label of malware vs non-malware, where 1 is malware and 0 is benign
y = df.iloc[:,3].values

# print("x:", x, "\ny:", y)
# this splits the data. So the settings for the tutorial use a test size of 0.2, 
# which means 20% of the data will be used for testing, while 80% is used for training.
# and 42 is just the random seed used for what data is used for training and testing
# if I set the test size to 0.9, then it has a few false negatives
# but otherwise it just stays at 100% accuracy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# random forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

feature_importances = classifier.feature_importances_

# showing the importance of features
plt.barh(x.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.show()

# metric scores accuracy, F1 score, AUC-ROC, Average Precision (mAP), or domain-specific metrics

f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)
prec_score = average_precision_score(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1 score:", f1)
print("AUC ROC curve value:", auc_roc)
print("Average precision score:", prec_score)