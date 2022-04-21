import json
import numpy as np


def concat_int(intents):
    """
    Fonction utilitaire permettant de gérer les clés des dictionaires d'intents
    """
    res = ''
    for i in intents:
        res += i + '+'
    return res



def load_data(type, n):
    """
    Charge les données
    :param type: type des data
    :param n: nombre de fichiers de data
    :return: utterances -> liste de dialogue, un dialogue est une liste de phrases
            intents -> liste de liste de labels : (speaker, intent) un intent est une liste d'intent
            label_combinations -> dictionnaire d'occurences des intents
    """

    utterances = []
    intents = []
    label_combinations = dict()
    for i in range(1,n+1):
        f = open(f"data/{type}/dialogues_{str(i).zfill(3)}.json")
        data = json.load(f)
        for dialog in data:
            turns = dialog['turns']
            dialog_utt = []
            dialog_int = []
            for t in turns:
                actions = t['frames'][0]['actions']
                acts = []
                for a in actions:
                    act = a['act']
                    if act not in acts:
                        acts.append(act)
                acts_concat = concat_int(acts)
                if acts_concat not in label_combinations:
                    label_combinations[acts_concat] = 1
                else :
                    label_combinations[acts_concat] += 1
                speaker = t['speaker']
                dialog_int.append((speaker, acts))
                utterance = t['utterance']
                dialog_utt.append(utterance)
            utterances.append(dialog_utt)
            intents.append(dialog_int)

    return utterances, intents, label_combinations


def label_preprocessing_greetings(intents):
    """
    Supprime les intent de greetings quand ils sont accompagnés d'autres intents
    :param intents: liste de liste de labels
    :return: liste des intents modifiée
    """
    for d_int in intents:
        for speaker, intent in d_int:
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
        for speaker, intent in d:
            if concat_int(intent) not in intents_to_keep:
                alea = np.random.randint(0, len(intent))
                intent = intent[alea]
                intent_mod.append((speaker, intent))
        res.append(intent_mod)
    return res


def label_dict_occ(labels):
    """
    Crée le dictionnaire d'occurences des labels
    """
    labels_dict = dict()
    for d in labels:
        for speaker, label in d:
            if type(label) == list:
                labels_concat = concat_int(label)
            else:
                labels_concat = label
            if labels_concat not in labels_dict:
                    labels_dict[labels_concat] = 1
            else:
                labels_dict[labels_concat] += 1
    return labels_dict






