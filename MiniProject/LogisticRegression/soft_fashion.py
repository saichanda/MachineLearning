'''
                         Aum Sri Sai Ram
    In this code, we show how 10 logistic regression learners can be used for 
    digit classification.
'''    
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
''' Load the data'''
X = np.array(pd.read_csv('fashion-mnist_train.csv'))
y_train = X[:,:1]
X_train = X[:,1:]
X_t = np.array(pd.read_csv('fashion-mnist_test.csv'))
y_test = X_t[:,:1]
X_test = X_t[:,1:]

logistic = LogisticRegression(multi_class='multinomial',solver='lbfgs')
logistic.fit(X_train,y_train)
predicted = logistic.predict(X_test)
print (metrics.accuracy_score(predicted,y_test))
import tensorflow as tf
from keras.utils import to_categorical

y_test = to_categorical(np.array(y_test.ravel()))
predicted = to_categorical(np.array(predicted.ravel()))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test, predicted, pos_label = None)
    roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), predicted.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for fashion_mnist')
plt.legend(loc="lower right")
plt.savefig('ROC.png')


target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted, target_names=target_names))
