from question_processing.main_question_processing import question_processing
from passage_retrival.main_passage_retrieval import passage_retrieval
from answer_processing.main_answer_processing import answer_processing_2, answer_processing_1
from question_processing.question_classifier_nn import Net2, Question_classifier_nn


train_path = "./data/dataframes/train.csv"
test_path = "./data/dataframes/test.csv"
eval_path = "./data/dataframes/eval.csv"  
modelPath = './data/mlp_model/test.pth'


### files for the evaluation
question_eval_path = './data/dataframes/eval.csv'
result_validate_path = './data/eval/our_results/miler.csv'


### files of the given questions and the results
given_questions = './data/given_questions.txt'
result_path = './data/result.csv'

### params for the classifier model
hidden_size = 500
epochs = 100
learning_rate = 0.0001
####


def train_classifier():
    '''trains classifier and returns it'''
    qc = Question_classifier_nn(Net2, train_path, test_path, hidden_size, epochs, learning_rate)
    #qc.load_model(modelPath)   
    qc.run_train_and_test()
    qc.save_model(modelPath)
    return qc

def load_classifier():
    '''loads classifier and returns it'''
    qc = Question_classifier_nn(Net2, train_path, test_path, hidden_size, epochs, learning_rate)
    qc.load_model(modelPath)
    return qc

def get_answers_to_question(question, qc):
    ''' finds an answer to the given question '''
    print("----------------------------------")
    print("question:", question)
    question_type, question_words, keywords = question_processing(question, qc)
    print("question_type: ", question_type)
    documents_sentences = passage_retrieval(question, min_similarity=0.45, size=50)
    # top_result = answer_processing_1(documents_sentences, question_words, question_type, keywords)
    # print("1", top_result)
    answers = answer_processing_2(question_type,documents_sentences, keywords, window=3)
    print("answer: ", answers[0] if len(answers) > 0 else "no answer found")
    return answers

def eval(qc):
    ''' for evaluation of the q and a system '''
    ### needs the given eval folder in data to work 
    sentences = []
    with open(question_eval_path) as file:
        for question in file:
            question = question.split(";")[1]
            question = question.replace("\n","")
            answers = get_answers_to_question(question, qc)
            text = ""
            for answer in answers:
                text += answer.word
                text += ";"
            text.strip(";")
            text += "\n"
            sentences.append(text)

    f = open(result_validate_path, 'a')
    
    for sentence in sentences:
        f.write(sentence)

def generate_results_from_questions(qc):
    ''' finds the answers to the given file '''
    sentences = []

    with open(given_questions) as file:
        for question in file:
            question = question.replace("\n","")
            answers = get_answers_to_question(question, qc)
            text = ""
            for answer in answers:
                text += answer.word
                text += ";"
            text.strip(";")
            text += "\n"
            sentences.append(text)

    f = open(result_path, 'a')
    for sentence in sentences:
        f.write(sentence)

def run():
    # qc = train_classifier() #wenn man den classifier neu Trainieren will
    qc = load_classifier() #wenn man einen trainierten classifier laden will
    generate_results_from_questions(qc)


if __name__ == '__main__':
    run()