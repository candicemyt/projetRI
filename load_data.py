#%%
import json
import matplotlib.pyplot as plt
#%%

def concat_int(intents):
    res = ''
    for i in intents:
        res += i + '+'
    return res



def load_data(type, n):
    """
    :param type: type des data
    :param n: nombres de fichiers de data
    :return: utterances -> liste de dialogue, un dialogue est une liste de phrases
            intents -> liste de liste de labels : (speaker, intent) un intent est une liste d'intent
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
xt, yt, l = load_data("train",127)
print(len(l))


#%%
l = dict(sorted(l.items(), key=lambda item: item[1]))
s = sum(list(l.values()))
v = [v/s for v in list(l.values())]
plt.bar(l.keys(), v)
plt.xticks(list(l.keys()), rotation=90, size=8)
print(l)
#%%
