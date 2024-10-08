import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r'S:\Data Science Prakash Senapati\AllDatasets\pulsar_stars.csv')

df.shape

df.head()

df.columns

df.columns=df.columns.str.strip()
df.columns

df.columns=['IP Mean','IP Sd','IP Kurtosis','IP Skewness','DM-SNR Mean','DM-SNR Sd','DM-SNR Kurtosis','DM-SNR Skewness','target_class']

df.columns

df['target_class'].value_counts()

df.info()

df.isnull().sum()

round(df.describe(),2)

plt.figure(figsize=(24,20))

plt.subplot(4,2,1)
fig=df.boxplot(column='IP Mean')
fig.set_title('')
fig.set_ylabel('IP Mean')

plt.subplot(4,2,2)
fig=df.boxplot(column='IP Sd')
fig.set_title('')
fig.set_ylabel('IP Sd')

plt.subplot(4,2,3)
fig=df.boxplot(column='IP Kurtosis')
fig.set_title('')
fig.set_ylabel('IP Kurtosis')

plt.subplot(4,2,4)
fig=df.boxplot(column='IP Skewness')
fig.set_title('')
fig.set_ylabel('IP Skewness')

plt.subplot(4,2,5)
fig=df.boxplot(column='DM-SNR Mean')
fig.set_title('')
fig.set_ylabel('DM-SNR Mean')

plt.subplot(4,2,6)
fig=df.boxplot(column='DM-SNR Sd')
fig.set_title('')
fig.set_ylabel('DM-SNR Sd')

plt.subplot(4,2,7)
fig=df.boxplot(column='DM-SNR Kurtosis')
fig.set_title('')
fig.set_ylabel('DM-SNR Kurtosis')

plt.subplot(4,2,8)
fig=df.boxplot(column='DM-SNR Skewness')
fig.set_title('')
fig.set_ylabel('DM-SNR Skewness')

plt.figure(figsize=(24,20))

plt.subplot(4, 2, 1)
fig = df['IP Mean'].hist(bins=20)
fig.set_xlabel('IP Mean')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 2)
fig = df['IP Sd'].hist(bins=20)
fig.set_xlabel('IP Sd')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 3)
fig = df['IP Kurtosis'].hist(bins=20)
fig.set_xlabel('IP Kurtosis')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 4)
fig = df['IP Skewness'].hist(bins=20)
fig.set_xlabel('IP Skewness')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 5)
fig = df['DM-SNR Mean'].hist(bins=20)
fig.set_xlabel('DM-SNR Mean')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 6)
fig = df['DM-SNR Sd'].hist(bins=20)
fig.set_xlabel('DM-SNR Sd')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 7)
fig = df['DM-SNR Kurtosis'].hist(bins=20)
fig.set_xlabel('DM-SNR Kurtosis')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 8)
fig = df['DM-SNR Skewness'].hist(bins=20)
fig.set_xlabel('DM-SNR Skewness')
fig.set_ylabel('Number of pulsar stars')

x=df.drop(['target_class'],axis=1)
y=df['target_class']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape,x_test.shape

cols=x_train.columns

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=pd.DataFrame(x_train,columns=[cols])
x_test=pd.DataFrame(x_test,columns=[cols])
x_train.describe()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc_c_1=SVC()
svc_c_1.fit(x_train,y_train)
y_pred=svc_c_1.predict(x_test)
print('Model accuracy score with defult hyperparameters : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

svc_c_100=SVC(C=100.0) 
svc_c_100.fit(x_train,y_train)
y_pred=svc_c_100.predict(x_test)
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

svc_c_1000=SVC(C=1000.0) 
svc_c_1000.fit(x_train,y_train)
y_pred=svc_c_1000.predict(x_test)
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

svc_l_1=SVC(kernel='linear', C=1.0) 
svc_l_1.fit(x_train,y_train)
y_pred=svc_l_1.predict(x_test)
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

svc_l_100=SVC(kernel='linear', C=100.0) 
svc_l_100.fit(x_train,y_train)
y_pred=svc_l_100.predict(x_test)
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

svc_l_1000=SVC(kernel='linear', C=1000.0) 
svc_l_1000.fit(x_train,y_train)
y_pred=svc_l_1000.predict(x_test)
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train=svc_l_1.predict(x_train)
print(y_pred_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train,y_pred_train)))
print('Training set score:{0:0.4f}'.format(svc_c_1.score(x_test,y_test)))

y_test.value_counts()

null_accuracy=(3306/(3306+274))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

poly_svc=SVC(kernel='poly',C=1.0)
poly_svc.fit(x_train,y_train)
y_pred=poly_svc.predict(x_test)
print('Model accuracy score with polynomial kernel and c=1.0 : {0:0.4}'. format(accuracy_score(y_test,y_pred)))

poly_svc_100=SVC(kernel='poly',C=100.0)
poly_svc_100.fit(x_train,y_train)
y_pred=poly_svc_100.predict(x_test)
print('Model accuracy score with polynomial kernel and c=100.0 : {0:0.4}'. format(accuracy_score(y_test,y_pred)))

poly_svc_1000=SVC(kernel='poly',C=1000.0)
poly_svc_1000.fit(x_train,y_train)
y_pred=poly_svc_1000.predict(x_test)
print('Model accuracy score with polynomial kernel and c=1000.0 : {0:0.4}'. format(accuracy_score(y_test,y_pred)))

sigmoid_svc_1=SVC(kernel='sigmoid',C=1.0)
sigmoid_svc_1.fit(x_train,y_train)
y_pred=sigmoid_svc_1.predict(x_test)
print('Model accuracy score with polynomial kernel and c=1000.0 : {0:0.4}'. format(accuracy_score(y_test,y_pred)))

sigmoid_svc_10=SVC(kernel='sigmoid',C=10.0)
sigmoid_svc_10.fit(x_train,y_train)
y_pred=sigmoid_svc_10.predict(x_test)
print('Model accuracy score with Sigmoid kernel and c=1.0 : {0:0.4}'. format(accuracy_score(y_test,y_pred)))

sigmoid_svc_100=SVC(kernel='sigmoid',C=100.0)
sigmoid_svc_100.fit(x_train,y_train)
y_pred=sigmoid_svc_100.predict(x_test)
print('Model accuracy score with Sigmoid kernel and c=100.0 : {0:0.4}'. format(accuracy_score(y_test,y_pred)))

sigmoid_svc_1000=SVC(kernel='sigmoid',C=1000.0)
sigmoid_svc_1000.fit(x_train,y_train)
y_pred=sigmoid_svc_1000.predict(x_test)
print('Model accuracy score with Sigmoid kernel and c=1000.0 : {0:0.4}'. format(accuracy_score(y_test,y_pred)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test))

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))

true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))

import pickle

# Saving the SVM models

# Save the default SVC model
with open('svc_default_model.pkl', 'wb') as file:
    pickle.dump(svc_c_1, file)

# Save the SVC model with rbf kernel and C = 100.0
with open('svc_rbf_100_model.pkl', 'wb') as file:
    pickle.dump(svc_c_100, file)

# Save the SVC model with rbf kernel and C = 1000.0
with open('svc_rbf_1000_model.pkl', 'wb') as file:
    pickle.dump(svc_c_1000, file)

# Save the SVC model with linear kernel and C = 1.0
with open('svc_linear_1_model.pkl', 'wb') as file:
    pickle.dump(svc_l_1, file)

# Save the SVC model with linear kernel and C = 100.0
with open('svc_linear_100_model.pkl', 'wb') as file:
    pickle.dump(svc_l_100, file)

# Save the SVC model with linear kernel and C = 1000.0
with open('svc_linear_1000_model.pkl', 'wb') as file:
    pickle.dump(svc_l_1000, file)

# Save the SVC model with polynomial kernel and C = 1.0
with open('poly_svc_1_model.pkl', 'wb') as file:
    pickle.dump(poly_svc, file)

# Save the SVC model with polynomial kernel and C = 100.0
with open('poly_svc_100_model.pkl', 'wb') as file:
    pickle.dump(poly_svc_100, file)

# Save the SVC model with sigmoid kernel and C = 1.0
with open('svc_sigmoid_1_model.pkl', 'wb') as file:
    pickle.dump(sigmoid_svc_1, file)

# Save the SVC model with sigmoid kernel and C = 100.0
with open('svc_sigmoid_100_model.pkl', 'wb') as file:
    pickle.dump(sigmoid_svc_100, file)