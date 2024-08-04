from question_processing.question_classifier_nn import Question_classifier_nn, Net2
from question_processing.text_processing import text_processing

def get_all_question_words(question):
    ''' returns the keywords, synonyms, hypernyms of the question '''
    text_processer = text_processing()
    words = []
    
    question_words = text_processer.string_to_words(question)
    key_words = text_processer.get_key_words(question_words)
    words += text_processer.get_synonyms_for_word_array(key_words) 
    words += text_processer.get_hypernyms_for_word_array(key_words)
    words += key_words

    return list(set(words)), key_words

def get_question_type(question, model):
    ''' returns the question type of the question '''
    question_words = ["when", "where" ,"who", "how many", "year"]
    question_types = ["NUM:date","LOC:country","HUM:ind", "NUM:count", "NUM:date"]
    matched = False
    for question_word in question_words:
        if question_word in question.lower():
            question_type = question_types[question_words.index(question_word)]
            matched = True
            break
    
    if not matched:
        question_type = model.predict_question(question)

    return question_type

def question_processing(question, model):
    ''' returns the question type, the words and the keywords of the question '''

    question_type = get_question_type(question, model)

    words, keywords = get_all_question_words(question)

    return question_type, words, keywords