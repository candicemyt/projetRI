import pandas as pd
import numpy as np
from load_data import *
from preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def content_feature(utterances):
    #tf-idf representation
    vectorizer = TfidfVectorizer()
    #todo nom de variable significatif pour ret
    ret = vectorizer.fit_transform(utterances)

    cos_initial_utterance = []
    cos_entire_dialogue = []
    question = []
    same = []
    wh = []

    wh_words = ['how', 'what', 'where', 'who', 'when', 'why']
    for i in range(len(utterances)):
        cos_initial_utterance.append(cosine_similarity(ret[i], ret[0]))
        cos_entire_dialogue.append(np.sum(cosine_similarity(ret[i], ret)))
        if '?' in utterances[i]:
            question.append(1)
        else:
            question.append(0)

        if 'same' in utterances[i] or 'similar' in utterances[i]:
            same.append(1)
        else:
            same.append(0)

        is_wh = False
        for wh_word in wh_words:
            if wh_word in utterances[i]:
                wh.append(1)
                is_wh = True
                break
        if not is_wh:
            wh.append(0)

    cos_initial_utterance = np.array(cos_initial_utterance).reshape(len(utterances))
    return list(zip(cos_initial_utterance, np.array(cos_entire_dialogue), question, same, wh))


def structural_feature(utterances):

    pos = np.arange(0, len(utterances))
    nor_pos = []
    utterances_wo_sw = []
    num_w = []
    sw = []
    sw_stem = []
    is_starter = []

    for i in range(len(utterances)):
        nor_pos.append(round((i/len(utterances)), 3))
        utterances_wo_sw = stop_words_removal(utterances[i])
        num_w.append(len(utterances_wo_sw.split(' ')))
        sw.append(len(set(utterances_wo_sw.split(' '))))
        utterances_preprocessed = preprocessing(utterances_wo_sw)
        sw_stem.append(len(set(utterances_preprocessed.split(' '))))
        if i%2 == 0:
            is_starter.append(1)
        else:
            is_starter.append(0)

    return list(zip(pos, nor_pos, num_w, sw, sw_stem, is_starter))


def sentiment_feature(utterances):

    thank = []
    exc = []
    neg = []
    sent = []
    sid = SentimentIntensityAnalyzer()

    for i in range(len(utterances)):

        if 'thank' in utterances[i]:
            thank.append(1)
        else:
            thank.append(0)

        if '!' in utterances[i]:
            exc.append(1)
        else:
            exc.append(0)

        if 'did not' in utterances[i] or 'does not' in utterances[i]:
            neg.append(1)
        else:
            neg.append(0)

        sent.append(list(map(str, [sid.polarity_scores(utterances[i])['neg'], sid.polarity_scores(utterances[i])['neu'],
                                   sid.polarity_scores(utterances[i])['pos']])))

    return list(zip(thank, exc, neg, sent))


def feature_analysis(dialogs):
    features = []
    #i = 0
    for diag in dialogs:
        #print(i, "/", len(dialogs))
        #i += 1
        content = [list(i) for i in content_feature(diag)]
        structural = [list(i) for i in structural_feature(diag)]
        sentiment = [list(i) for i in sentiment_feature(diag)]
        feature = [list(i) for i in list(zip(content, structural, sentiment))]
        features.append(feature)
    return features


utterances, intents, _ = load_data("train", 127)
# %%
utterances_preprocess = [[preprocessing(text) for text in dialogue] for dialogue in utterances]
# %%
features = feature_analysis(utterances_preprocess[:10])
diag = pd.DataFrame(features[0])




