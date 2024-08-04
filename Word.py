
import nltk
import spacy
from nltk.corpus import wordnet as wn
nlp = spacy.load("en_core_web_sm")
stem = nltk.stem.PorterStemmer()

class Word():
    def __init__(self, word):
        self.word = stem.stem(word)
    
    def __str__(self) -> str:
        return self.word

    def __repr__(self) -> str:
        return self.word

    def __eq__(self, __o) -> bool:
        if isinstance(__o, str):
            return self.word == __o
        return self.word == __o.word
    
    def __hash__(self) -> int:
        return hash(self.word)

    def get_type(self):
        ''' returns the type of the word '''
        doc = nlp(self.word)
        for token in doc:
            return token.pos_
            
    def get_tags(self):
        return nltk.pos_tag([self.word])[0][1]


    def get_ne(self):
        ''' returns the named entity of the word '''
        doc = nlp(self.word)
        for entity in doc.ents:
            return entity.label_

    def get_synonyms(self):
        ''' returns all synonyms of the word '''
        synsets = wn.synsets(self.word)
        synonyms = set()

        for syn in synsets:
            for i in syn.lemmas():
                if i.name() != self.word:
                    synonyms.add(Word(i.name().replace("_", " ")))
                
        return list(synonyms)

    def get_hypernyms(self):
        ''' returns all hypernyms of the word '''
        synsets = wn.synsets(self.word)
        
        if not synsets:
            return []
        s = synsets[0]
        hypernyms = []
        
        if not s.hypernyms():
            return []
        hyper = s.hypernyms()[0]
        hypernyms.append(Word(hyper.lemma_names()[0]))
        if not hyper.hypernyms():
            return hypernyms

        while hyper.lemma_names()[0] != "entity" and hyper.hypernyms():
            hyper = hyper.hypernyms()[0]
            hypernyms.append(Word(hyper.lemma_names()[0].replace("_", " ")))
        
        return hypernyms[:3 if len(hypernyms) > 3 else len(hypernyms)]

