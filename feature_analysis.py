import numpy as np
import pandas as pd

from load_data import *
from preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def content_feature(utterances):
    #tf-idf representation
    vectorizer = TfidfVectorizer()
    ret = vectorizer.fit_transform(utterances)
    #Cosine similarity between the utterance and the first utterance of the dialog
    cos_initial_utterance = [cosine_similarity(ret[i], ret[0]) for i in range(len(utterances))]
    cos_initial_utterance = np.array(cos_initial_utterance).reshape(len(utterances))
    #Cosine similarity between the utterance and the entire dialog
    cos_entire_dialogue = np.array([np.sum(cosine_similarity(ret[i], ret)) for i in range(len(utterances))])
    #Does the utterance contain a question mark
    question=[1 if s.find('?') != -1 else 0 for s in utterances]
    #Does the utterance contain the keywords same, similar
    same=[1 if s.find('same') != -1 or s.find('similar') != -1 else 0 for s in utterances]
    #Does the utterance contain the keywords what, where, when, why, who, how
    wh = [[1 if s.find('how') != -1 else 0,1 if s.find('what') != -1 else 0,1 if s.find('why') != -1 else 0,1 if s.find('who') != -1 else 0,1 if s.find('where') != -1 else 0,1 if s.find('when') != -1 else 0] for s in utterances]
    return pd.DataFrame(list(zip(cos_initial_utterance,cos_entire_dialogue,question,same,wh)), columns = ['cos_initial_utterance', 'cos_entire_dialogue','question_mark','same/similar','how/what/why/who/where/when'])

def structural_feature(utterances):
    #Absolute position of an utterance in the dialog
    pos=[i for i in range(len(utterances))]
    #Normalized position of an utterance in the dialog (AbsPos divided by # utterances)
    nor_pos=[i / len(utterances) for i in range(len(utterances))]
    #Total number of words in an utterance after stop words removal
    utterances=[stop_words_removal(text) for text in utterances]
    num_w=[len(s.split(' ')) for s in utterances]
    #Unique number of words in an utterance after stop words removal
    sw=[len(set(s.split(' '))) for s in utterances]
    #Unique number of words in an utterance after stop words removal and stemming
    utterances=[preprocessing(text) for text in utterances]
    sw_stem=[len(set(s.split(' '))) for s in utterances]
    return pd.DataFrame(list(zip(pos,nor_pos,num_w,sw,sw_stem)), columns = ['position','normalized_position','nb_word_sw','unique_nb_word_sw','unique_nb_word_sw_stem'])


def sentiment_feature(utterances):
    #Does the utterance contain the keyword thank
    thank=[1 if s.find('thank') != -1 else 0 for s in utterances]
    #Does the utterance contain an exclamation mark
    exc=[1 if s.find('!') != -1 else 0 for s in utterances]
    #Does the utterance contain the keyword did not, does not
    neg=[1 if s.find('did not') != -1 or s.find('does not') != -1 else 0 for s in utterances]
    #Sentiment scores of the utterance computed by VADER (positive, neutral, and negative)
    sid = SentimentIntensityAnalyzer()
    sent=[list(map(str, [sid.polarity_scores(s)['neg'], sid.polarity_scores(s)['neu'], sid.polarity_scores(s)['pos']])) for s in utterances]
    return pd.DataFrame(list(zip(thank,exc,neg,sent)), columns = ['thank','exclamation','did_not/does_not','pos/neu/neg'])


def feature_analysis(dialogs):
    features=[]
    for diag in dialogs:
        content= content_feature(diag)
        structural= structural_feature(diag)
        sentiment= sentiment_feature(diag)
        feature=content.join(structural)
        feature=feature.join(sentiment)
        features.append(feature)
    return features

if __name__ == '__main__':

    xt, yt, l = load_data("train",127)
    xt_preprocess=[[preprocessing(text) for text in dialogue]for dialogue in xt]
    features=feature_analysis(xt_preprocess)
    diag=features[0]
    feature=feature_analysis(diag)
    print(feature.head())
