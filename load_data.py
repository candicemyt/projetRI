#%%
import json
import matplotlib.pyplot as plt
import numpy as np
#%%

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
        f = open(f"./data/{type}/dialogues_{str(i).zfill(3)}.json")
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

#%%
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

#%%
def label_preprocessing_combinations(intents, intents_to_keep):
    """

    :param intents:
    :param intents_to_keep:
    :return:
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


#%%
def label_dict_occ(labels):
    labels_dict = dict()
    for d in labels:
        for speaker, label in d:
            labels_concat = concat_int(label)
            if labels_concat not in labels_dict:
                labels_dict[labels_concat] = 1
            else:
                labels_dict[labels_concat] += 1
    return labels_dict


#%%
if __name__ == '__main__':

    utterances, labels, dico_labels = load_data("train",127)
    print(len(dico_labels))
    #%%
    ##plot histogramme labels avant preprocessing
    dico_labels = dict(sorted(dico_labels.items(), key=lambda item: item[1]))
    s = sum(list(dico_labels.values()))
    v = [v/s for v in list(dico_labels.values())]
    plt.bar(dico_labels.keys(), v)
    plt.xticks(list(dico_labels.keys()), rotation=90, size=8)

#%%
    ##premier preprocessing
    labels_preprocessed = label_preprocessing_greetings(labels)
    dico_labels_preprocessed = label_dict_occ(labels_preprocessed)
    print(len(dico_labels_preprocessed))
#%%
    ##plot histogramme cumulatif des frequences des labels apres premier preprocessing
    dico_labels_preprocessed = dict(sorted(dico_labels_preprocessed.items(), key=lambda item: item[1], reverse=True))
    s = sum(list(dico_labels_preprocessed.values()))
    v = [v / s for v in list(dico_labels_preprocessed.values())]
    v_cum = np.cumsum(v)
    plt.bar(dico_labels_preprocessed.keys(), v_cum)
    plt.xticks(list(dico_labels_preprocessed.keys()), rotation=90, size=8)


#%%
    #recherche des labels qui représentent 90% des labels
    for i in range(len(v_cum)):
        if v_cum[i]>0.9:
            break
    intents_to_keep = list(dico_labels_preprocessed.keys())[0:i]

    ##deuxieme preprocessing
    labels_preprocessed = label_preprocessing_combinations(labels_preprocessed, intents_to_keep)
    dico_labels_preprocessed = label_dict_occ(labels_preprocessed)
    print(len(dico_labels_preprocessed))
#%%
    ##plot histogramme des frequences des labels apres second preprocessing
    dico_labels = dict(sorted(dico_labels_preprocessed.items(), key=lambda item: item[1]))
    s = sum(list(dico_labels.values()))
    v = [v/s for v in list(dico_labels.values())]
    plt.bar(dico_labels.keys(), v)
    plt.xticks(list(dico_labels.keys()), rotation=90, size=8)
#%%
