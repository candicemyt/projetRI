import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords

sw=set(stopwords.words('english'))

def preprocessing(text, low, punc, up_word, number, stem, stw, m='°'):
    # suppression des accents et des caractères non normalisés
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

    # transformation des mots entièrement en majuscule en marqueurs spécifiques
    # ici on s'intéresse aux mots de taille supérieure ou égale à 3
    if up_word:
        text = re.sub(r'([A-Z_0-9]{3,}[\s\n])', m + ' ', text)

    # transformation en minuscule
    if low:
        text = text.lower()

    # suppression de la ponctuation
    if punc:
        punc = string.punctuation
        punc += '\n\r\t'
        text = text.translate(str.maketrans(punc, ' ' * len(punc)))

    # suppression des chiffres
    if number:
        text = re.sub('[0-9]+', '', text)

    # on enlève les stop words
    if stw:
        text = ' '.join([word for word in text.split() if word not in sw])

    # stemming
    if stem:
        s = nltk.stem.SnowballStemmer('english')
        text = ' '.join([s.stem(word) for word in text.split()])

    return text