import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance

from feature_analysis import feature_analysis
from intents_preprocess import intents_preprocessing
from load_data import *
from preprocessing import preprocessing, stop_words_removal, stemming
#%%

# todo : il faut donner en datax aux modèles les features générées et pas les phrases voir ce qu'il faut donner pour
#  les datay (est ce que binary relevance suffit?)


X_train, labels_train, dico_labels = load_data("train",127)
X_val, labels_val, dico_labels_val = load_data("val",20)
X_test, labels_test,dico_labels_test = load_data("test",34)

#%%
##preprocessing train
labels_preprocessed_train =intents_preprocessing(labels_train)
X_train_preprocess=[[preprocessing(text) for text in dialogue]for dialogue in X_train]
features_train=feature_analysis(X_train_preprocess)
#%%
##preprocessing val
labels_preprocessed_val =intents_preprocessing(labels_val)
X_val_preprocess=[[preprocessing(text) for text in dialogue]for dialogue in X_val]
features_val=feature_analysis(X_val_preprocess)
#%%
##preprocessing test
labels_preprocessed_test =intents_preprocessing(labels_test)
X_test_preprocess=[[preprocessing(text) for text in dialogue]for dialogue in X_test]
features_test=feature_analysis(X_test_preprocess)
#%%
# todo: essayer avec GaussianNB()
clf = BinaryRelevance(MultinomialNB())

#%%
##we use train set to fit the model
clf.fit(features_train, labels_preprocessed_train)

#%%
##we use validation set to fine-tune the model hyperparameters with CalibratedClassifierCV
#CalibratedClassifierCV : estimate the parameters of a classifier and calibrate the classifier
#we use binaryRelevance()
cal_clf = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
print(cal_clf.get_params())
binary_rel_clf = BinaryRelevance(cal_clf)
binary_rel_clf.fit(features_val,labels_preprocessed_val)

#%%
##we use Test Dataset to evaluate the model
pred = clf.predict(features_test)
acc=accuracy_score(labels_preprocessed_test,pred)
pre=precision_score(labels_preprocessed_test,pred)
rec=recall_score(labels_preprocessed_test,pred)
f1=f1_score(labels_preprocessed_test,pred)
df=pd.DataFrame(['NaiveBayes',acc,pre,rec,f1],columns=['methods', 'acc', 'precision','recall','f1'])
print(df)
#%%
