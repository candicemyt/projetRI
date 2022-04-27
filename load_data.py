import json

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
            intents -> liste de liste de labels : (intent) un intent est une liste d'intent
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
                dialog_int.append(acts)
                utterance = t['utterance']
                dialog_utt.append(utterance)
            utterances.append(dialog_utt)
            intents.append(dialog_int)

    return utterances, intents, label_combinations







