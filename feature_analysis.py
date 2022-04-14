from load_data import *
from preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

xt, yt, l = load_data("train",127)

def compute_tf_idf(xt):
    vectorizer = TfidfVectorizer()
    ret = vectorizer.fit_transform(xt)
    return ret

xt_preprocess=[[preprocessing(text, True, True, True, True, True,True,'°') for text in dialogue]for dialogue in xt]
print(len(xt_preprocess[0]))
tf_idf=compute_tf_idf(xt_preprocess[0])
print(tf_idf)
