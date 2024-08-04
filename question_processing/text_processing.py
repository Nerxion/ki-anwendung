import nltk, re
import spacy
from nltk.corpus import wordnet as wn

from Word import Word


class text_processing():
    ''' class for processing text '''
    def __init__(self):
        self.stem = nltk.stem.PorterStemmer()
        self.nlp = spacy.load("en_core_web_sm")


    def get_synonyms_for_word_array(self,arr):
        ''' returns synonyms for all words of a list of words '''
        syns = []
        for word in arr:
            syns += word.get_synonyms()

        return syns
    
    def get_hypernyms_for_word_array(self,arr):
        ''' returns hypernyms for all words of a list of words '''
        hyper = []
        for word in arr:
            hyper += word.get_hypernyms()

        return hyper

    def string_to_words(self, text:str):
        ''' returns converted text to list of words '''
        textarr = nltk.word_tokenize(text)
        textarr2 = []
        for i, word in enumerate(textarr):
            if i == 0:
                textarr2.append(word)
            else:
                if word.istitle() and textarr[i-1].istitle():
                    textarr2[-1] += ' ' + word
                elif word.isnumeric() and textarr[i-1].istitle():
                    textarr2[-1] += ' ' + word
                else:
                    textarr2.append(word)
        
        linearr = [Word(ele) for ele in textarr2 if not re.compile(r'[^a-zA-Z0-9 ]+').match(ele)] # für jedes Wort checken, dass es kein Sonderzeichen/Zahl, dann in lowerform in die Liste hinzufügen
        return linearr

    def get_key_words(self, words:list):
        ''' returns keywords of a list of words '''
        unwanted_tags = ["CC", "DT", "EX", "IN", "MD"]

        return [word for word in words if word.get_tags() not in unwanted_tags]
