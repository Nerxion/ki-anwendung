from answer_processing.name_entities import get_named_entities
import fasttext
import fasttext.util
from numpy import dot
from numpy.linalg import norm
# from name_entities import get_named_entities


def convert_question_to_answer_type(question_type):
    ''' returns the correct class of the NE depending on the question class '''
    categories = ["HUM:ind", "LOC:other", "NUM:count", "NUM:date", "ENTY:other", "ENTY:cremat", "HUM:gr", "LOC:country", "LOC:city", "ENTY:animal", "ENTY:food"]
    ne_classes = ["PERSON", "LOC", "CARDINAL", "DATE", "PRODUCT" ,"WORK_OF_ART", "NORP", "ORG", "GPE", "GPE", "PERSON", "PRODUCT"]
    
    ne_class = ne_classes[categories.index(question_type)]
    return ne_class

def get_potential_answers(ne, question_type):
    ''' returns the correct class of the NE depending on the question class '''
    ne_class = convert_question_to_answer_type(question_type)
    return ne[ne_class]


def find_answer(potential_answers, keywords):
    ''' counts how often an answer has been found and returns that as a dict, excluding words from the question '''
    answers = {}
    for match in potential_answers:
        if match not in keywords:
            if match not in answers.keys():
                answers[match] = 1
            else:
                answers[match] += 1
    return answers


def answer_processing_1(sentences, words, question_type, keywords):
    ''' old approach, returns the answer with the highest occurance '''
    ne = get_named_entities(sentences, words, keywords)
    potential_answers = get_potential_answers(ne, question_type)
    answer = find_answer(potential_answers, keywords)
    return answer
    #return max(answer, key=answer.get)

def border(x, y):
    ''' returns the modulo of the given number '''
    if x <= 0:
        return 0
    
    if x >= y:
        return y - 1
    else:
        return x
   
def find_answers(question_type, sentences, question_words, window):
    ''' finds all possible answers in the given sentences and returns them as a dict with the score'''
    answer_dict = dict()
    answer_type = convert_question_to_answer_type(question_type)

    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word not in question_words:
                continue
            
            for y in range(border(i-window, len(sentence)), border(i+window, len(sentence))):
                
                if sentence[y].get_ne() != answer_type:
                    continue

                if sentence[y] in question_words:
                    continue

                if sentence[y] in answer_dict.keys():
                    answer_dict[sentence[y]] += 1

                else:
                    answer_dict[sentence[y]] = 1
    return answer_dict

def answer_processing_2(question_type, sentences, question_words, window=4):
    ''' main function of the answer processing, returns the found answer which highest score '''
   
    answers = find_answers(question_type, sentences, question_words, window)
    answers = dict(sorted(answers.items(), key=lambda x: x[1], reverse=True))
    best = list(answers.keys())[:10 if len(answers) > 10 else len(answers)]

    return best
