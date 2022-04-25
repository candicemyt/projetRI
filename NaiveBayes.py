import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance
from load_data import load_data, label_preprocessing_greetings, label_preprocessing_combinations

#%%

#TO DO : essayer avec GaussianNB()

##data
from preprocessing import preprocessing, stop_words_removal, stemming

X_train, labels_train, dico_labels = load_data("train",127)
X_val, labels_val, dico_labels_val = load_data("val",20)
X_test, labels_test,dico_labels_test = load_data("test",34)

#%%
##preprocessing train
X_train =[[stemming(stop_words_removal(preprocessing(text))) for text in dialogue]for dialogue in X_train]
labels_preprocessed_train = label_preprocessing_greetings(labels_train)
intents_to_keep = ['REQUEST+', 'INFORM+', 'OFFER+', 'CONFIRM+', 'SELECT+', 'GOODBYE+', 'OFFER+INFORM_COUNT+', 'INFORM_INTENT+', 'INFORM+INFORM_INTENT+', 'AFFIRM+', 'NOTIFY_SUCCESS+', 'REQ_MORE+', 'OFFER_INTENT+', 'NEGATE+', 'THANK_YOU+', 'REQUEST+AFFIRM+', 'INFORM+NOTIFY_SUCCESS+']
Y_train = label_preprocessing_combinations(labels_preprocessed_train, intents_to_keep)
#%%
##preprocessing val
X_val =[[stemming(stop_words_removal(preprocessing(text))) for text in dialogue]for dialogue in X_val]
labels_preprocessed_val = label_preprocessing_greetings(labels_val)
Y_val = label_preprocessing_combinations(labels_preprocessed_val, intents_to_keep)
##preprocessing test
labels_preprocessed_test = label_preprocessing_greetings(labels_test)
Y_test = label_preprocessing_combinations(labels_preprocessed_test, intents_to_keep)

#%%
##we use train set to fit the model
clf=BinaryRelevance(MultinomialNB())

#%%
print(len(X_train))
print(len(Y_train))
print(len(X_train[0]))
print(len(Y_train[0]))
print(X_train[0][0])
print(Y_train[0])
#%%
clf.fit(X_train, Y_train)

#%%
##we use validation set to fine-tune the model hyperparameters with CalibratedClassifierCV
#CalibratedClassifierCV : estimate the parameters of a classifier and calibrate the classifier
#we use binaryRelevance()
cal_clf=CalibratedClassifierCV(base_estimator=clf, cv="prefit")
print(cal_clf.get_params())
binary_rel_clf=BinaryRelevance(cal_clf)
binary_rel_clf.fit(X_val,Y_val)

#%%
##we use Test Dataset to evaluate the model
pred = binary_rel_clf.predict(X_test)
acc=accuracy_score(Y_test,pred)
pre=precision_score(Y_test,pred)
rec=recall_score(Y_test,pred)
f1=f1_score(Y_test,pred)
df=pd.DataFrame(['NaiveBayes',acc,pre,rec,f1],columns=['methods', 'acc', 'precision','recall','f1'])
