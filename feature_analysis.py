#%%
import numpy as np
from load_data import *
from preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#%%
def compute_cosine_similarity(xt):
    vectorizer = TfidfVectorizer()
    ret = vectorizer.fit_transform(xt)
    cos_initial_utterance = [cosine_similarity(ret[i], ret[0])for i in range(len(xt))]
    cos_initial_utterance = np.array(cos_initial_utterance).reshape(len(xt))
    cos_entire_dialogue = [cosine_similarity(ret[i], ret)for i in range(len(xt))]
    cos_entire_dialogue = np.array(cos_entire_dialogue).reshape((len(xt),len(xt)))
    return cos_entire_dialogue

#%%
xt, yt, l = load_data("train",127)
xt_preprocess=[[preprocessing(text, True, True, True, True, True,True,'°') for text in dialogue]for dialogue in xt]
print(len(xt_preprocess[0]))

#%%
cos_sim=compute_cosine_similarity(xt_preprocess[0])
print(cos_sim.shape)
print(cos_sim)

