import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords

sw=set(stopwords.words('english'))

def preprocessing(text):
    # suppression des accents et des caractères non normalisés
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

    # transformation des mots entièrement en majuscule en marqueurs spécifiques
    # ici on s'intéresse aux mots de taille supérieure ou égale à 3
    text = re.sub(r'([A-Z_0-9]{3,}[\s\n])', '° ', text)

    # transformation en minuscule
    text = text.lower()

    # suppression des chiffres
    text = re.sub('[0-9]+', '', text)
    return text

def stop_words_removal(text):
    # on enlève les stop words
    text = ' '.join([word for word in text.split() if word not in sw])

    # suppression de la ponctuation
    punc = string.punctuation
    punc += '\n\r\t'
    text = text.translate(str.maketrans(punc, ' ' * len(punc)))
    return text

def stemming(text):
    s = nltk.stem.SnowballStemmer('english')
    text = ' '.join([s.stem(word) for word in text.split()])
    return text