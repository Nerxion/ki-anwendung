import re, nltk
from Word import Word
import numpy as np
from numpy import dot
from numpy.linalg import norm
from answer_processing.name_entities import get_named_entities
import fasttext
import fasttext.util

ft = fasttext.load_model('./data/fasttext/cc.en.300.bin')


def clean_document_sentences(document):
    ''' cleans all sentences of a document '''
    lines = document.split("\n")
    text = ""
    was_nicht = ["== External links ==", "== See also ==", "== References =="]
    filter = False
    for line in lines:
        if filter and line == "":
            filter = False
            continue

        if line in was_nicht:
            filter = True
            continue

        if line != "" and not line.startswith("==") and not filter:
            text += line + " "
    
    return text

def clean_documents(documents):
    ''' cleans a list of documents '''
    new_documents = []
    for document in documents:
        new_documents.append(clean_document_sentences(document))
    return new_documents

def filter_sentences_similarity(documents, question_sentence,min_similarity):
    ''' filters all sentences in a list of documents by similarity to the question '''
    sentences = dict()
    vec_question = ft.get_sentence_vector(question_sentence)
    for document in documents:
        for sentence in document:
            sentence = sentence.strip()
            if sentence == "":
                continue
            vec_sentence = ft.get_sentence_vector(sentence)
            cos_sim = dot(vec_question, vec_sentence)/(norm(vec_question)*norm(vec_sentence))
            
            if cos_sim < min_similarity:
                continue

            if cos_sim not in sentences.keys():
                sentences[cos_sim] = [sentence]
            else:
                sentences[cos_sim].append(sentence)
            
    sentences = dict(sorted(sentences.items(), reverse=True))
    return list(sentences.values())

def documents_to_sentences(documents):
    ''' splits a list of documents into a list of sentences '''
    new_documents = []
    for document in documents:
        # TODO: richtig in Saetze splitten
        #sentences = re.split(r'(?<!\w.\w.)(?<![A-Z][a-z].)(?<=.|?|!)\s', document)
        new_documents.append(document.split("."))
    return new_documents

def stem_sentences(sentences):
    ''' stems all words in a given list of sentences and returns them as a list '''
    new_sentences = []
    for sentence in sentences:
        for entry in sentence:
            new_sentences.append(stem_sentence(entry))
    return new_sentences

def stem_sentence(sentence):
    ''' stems all words in a given sentence and returns them as a list '''
    linearr = nltk.word_tokenize(sentence)

    linearr2 = []
    for i, word in enumerate(linearr):
        if i == 0:
            linearr2.append(word)
        else:
            if word.istitle() and linearr[i-1].istitle():
                linearr2[-1] += ' ' + word
            elif word.isnumeric() and linearr[i-1].istitle():
                linearr2[-1] += ' ' + word
            else:
                linearr2.append(word)

    linearr = [Word(ele) for ele in linearr2 if not re.compile(r'[^a-zA-Z0-9 ]+').match(ele)]
    return linearr
