import numpy as np
from load_data import *

def label_preprocessing_greetings(intents):
    """
    Supprime les intent de greetings quand ils sont accompagnés d'autres intents
    :param intents: liste de liste de labels
    :return: liste des intents modifiée
    """
    for d_int in intents:
        for intent in d_int:
            if len(intent)>=2:
                if 'GOODBYE' in intent and 'THANK_YOU' in intent:
                    if len(intent)==2:
                        break
                    else:
                        intent.remove('GOODBYE')
                        intent.remove('THANK_YOU')
                        break
                if 'GOODBYE' in intent:
                    intent.remove('GOODBYE')
                if 'THANK_YOU' in intent:
                    intent.remove('THANK_YOU')
    return intents


def label_preprocessing_combinations(intents, intents_to_keep):
    """
    Garde un seul intent aléatoirement dans les combinaisons d'intents trop peu fréquents
    :param intents: liste de listes d'intents
    :param intents_to_keep: liste des intents les plus fréquents
    :return: la liste de listes d'intents modifiée
    """
    res = []
    for d in intents:
        intent_mod = []
        for intent in d:
            if concat_int(intent) not in intents_to_keep:
                alea = np.random.randint(0, len(intent))
                intent = intent[alea]
                intent_mod.append([intent])
            else:
                intent_mod.append(intent)
        res.append(intent_mod)
    return res


def label_dict_occ(labels):
    """
    Crée le dictionnaire d'occurences des labels
    """
    labels_dict = dict()
    for d in labels:
        for label in d:
            if type(label) == list:
                labels_concat = concat_int(label)
            else:
                labels_concat = label
            if labels_concat not in labels_dict:
                labels_dict[labels_concat] = 1
            else:
                labels_dict[labels_concat] += 1
    return labels_dict


def intents_preprocessing(intents):
    """
    Agrège tous les preprocessing sur les intents
    :param intents: liste des intents fournie par load_data
    :return: la même liste preprocessed
    """
    #first intents preprocessing : greetings
    intents_preprocessed1 = label_preprocessing_greetings(intents)

    #searching labels which are representing 90% of labels
    dico_labels_preprocessed = label_dict_occ(intents_preprocessed1)
    dico_labels_preprocessed = dict(sorted(dico_labels_preprocessed.items(), key=lambda item: item[1], reverse=True))
    s = sum(list(dico_labels_preprocessed.values()))
    v = [v / s for v in list(dico_labels_preprocessed.values())]
    v_cum = np.cumsum(v)
    for i in range(len(v_cum)):
        if v_cum[i] > 0.9:
            break
    intents_to_keep = list(dico_labels_preprocessed.keys())[0:i]

    #second intents preprocessing : combinations
    intents_preprocessed2 = label_preprocessing_combinations(intents_preprocessed1, intents_to_keep)

    return intents_preprocessed2

