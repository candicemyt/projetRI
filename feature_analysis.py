#%%
import numpy as np
import pandas
from load_data import *
from preprocessing import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#%%
def content_feature(xt):
    #tf-idf representation
    vectorizer = TfidfVectorizer()
    ret = vectorizer.fit_transform(xt)
    #Cosine similarity between the utterance and the first utterance of the dialog
    cos_initial_utterance = [cosine_similarity(ret[i], ret[0])for i in range(len(xt))]
    cos_initial_utterance = np.array(cos_initial_utterance).reshape(len(xt))
    #Cosine similarity between the utterance and the entire dialog
    cos_entire_dialogue = np.array([np.sum(cosine_similarity(ret[i], ret))for i in range(len(xt))])
    #Does the utterance contain a question mark
    question=[1 if s.find('?') != -1 else 0 for s in xt ]
    #Does the utterance contain the keywords same, similar
    same=[1 if s.find('same') != -1 or s.find('similar') != -1 else 0 for s in xt ]
    #Does the utterance contain the keywords what, where, when, why, who, how
    wh = [[1 if s.find('how') != -1 else 0,1 if s.find('what') != -1 else 0,1 if s.find('why') != -1 else 0,1 if s.find('who') != -1 else 0,1 if s.find('where') != -1 else 0,1 if s.find('when') != -1 else 0] for s in xt]
    return list(zip(cos_initial_utterance,cos_entire_dialogue,question,same,wh))

#%%
def structural_feature(xt):
    #Absolute position of an utterance in the dialog
    pos=[i for i in range(len(xt))]
    #Normalized position of an utterance in the dialog (AbsPos divided by # utterances)
    nor_pos=[i/len(xt) for i in range(len(xt))]
    #Total number of words in an utterance after stop words removal
    xt=[preprocessing(text,low=False, punc=True, up_word=False, number=False, stem=False,stw=True,m='°') for text in xt]
    num_w=[len(s.split(' ')) for s in xt]
    #Unique number of words in an utterance after stop words removal
    sw=[len(set(s.split(' '))) for s in xt]
    #Unique number of words in an utterance after stop words removal and stemming
    xt=[preprocessing(text,low=False, punc=False, up_word=False, number=False, stem=True,stw=False,m='°') for text in xt]
    sw_stem=[len(set(s.split(' '))) for s in xt]
    return list(zip(pos,nor_pos,num_w,sw,sw_stem))

#%%
def sentiment_feature(xt):
    #Does the utterance contain the keyword thank
    thank=[1 if s.find('thank') != -1 else 0 for s in xt ]
    #Does the utterance contain an exclamation mark
    exc=[1 if s.find('!') != -1 else 0 for s in xt ]
    #Does the utterance contain the keyword did not, does not
    #Sentiment scores of the utterance computed by VADER (positive, neutral, and negative)
    #Number of positive and negative words from an opinion lexicon
    return list(zip(thank,exc))
#%%
xt, yt, l = load_data("train",127)

#%%
xt_preprocess=[[preprocessing(text, low=True, punc=False, up_word=True, number=True, stem=False, stw=False,m='°') for text in dialogue]for dialogue in xt]

#%%
content=content_feature(xt_preprocess[0])
df = pandas.DataFrame(content, columns = ['cos_initial_utterance', 'cos_entire_dialogue','question_mark','same/similar','how/what/why/who/where/when'])
print(df.head())

#%%
structural=structural_feature(xt_preprocess[0])
df = pandas.DataFrame(structural, columns = ['position','normalized_position','nb_word_sw','unique_nb_word_sw','unique_nb_word_sw_stem'])
print(df)

#%%
sentiment=sentiment_feature(xt_preprocess[0])
df = pandas.DataFrame(sentiment, columns = ['thank','exclamation'])
print(df)