#%%
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from load_data import load_data, label_preprocessing_greetings, label_dict_occ, label_preprocessing_combinations

parameters = [
    {
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.7, 1.0],
    },
    {
        'classifier': [SVC()],
        'classifier__kernel': ['rbf', 'linear'],
    },
]

#%%
X_train, labels, dico_labels = load_data("train",127)
labels_preprocessed = label_preprocessing_greetings(labels)
dico_labels_preprocessed = label_dict_occ(labels_preprocessed)
intents_to_keep = ['REQUEST+', 'INFORM+', 'OFFER+', 'CONFIRM+', 'SELECT+', 'GOODBYE+', 'OFFER+INFORM_COUNT+', 'INFORM_INTENT+', 'INFORM+INFORM_INTENT+', 'AFFIRM+', 'NOTIFY_SUCCESS+', 'REQ_MORE+', 'OFFER_INTENT+', 'NEGATE+', 'THANK_YOU+', 'REQUEST+AFFIRM+', 'INFORM+NOTIFY_SUCCESS+']
Y_train = label_preprocessing_combinations(labels_preprocessed, intents_to_keep)

#%%
clf = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy')
clf.fit(X_train, Y_train)

#%%
print (clf.best_params_, clf.best_score_)