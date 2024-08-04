from pathlib import Path
import nltk, re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle 
import heapq



class Question_classifier_logistic():
    '''classifier for questions'''
    def __init__(self):
        self.stem = nltk.stem.PorterStemmer()

    def build_bag_of_words(self, path):
        '''creating a vocab from data of the given path'''
        tokens = set()

        with open(path) as file:
            for line in file:
                line = line.split(";")

                linearr = nltk.word_tokenize(line[1])
                linearr = [ele.lower() for ele in linearr if not re.compile(r'[^a-zA-Z]+').match(ele)] # für jedes Wort checken, dass es kein Sonderzeichen/Zahl, dann in lowerform in die Liste hinzufügen

                for ele in linearr:
                    tokens.add(self.stem.stem(ele))

        print(len(tokens))
        self.vocab = list(tokens)

    def create_matrix(self, questions):
        '''creating a matrix with the quesions and the vocab'''
        tokens = self.vocab
        matrix = np.zeros(shape=(len(questions),len(tokens)))

        for i in range(len(questions)):
            for j in range(len(tokens)):
                if tokens[j] in questions[i]:
                    matrix[i][j] = 1
        return matrix

    def read_train_or_test_data(self, path):
        '''reads training or test data and returns anwsertyps and matrix'''
        df = pd.read_csv(path, sep = ";")
        questions = list(df["question"])
        answertype = list(df["answer_type"])
        matrix = self.create_matrix(questions)
        return matrix, answertype

    def pickeln(self, path, data):
        ''' pickels data
            to do: als eingene Klasse auslagern           
        '''
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        filepickle = open(path, "wb")
        pickle.dump(data, filepickle)
        filepickle.close()

    def logistic_aggression(self, matrix, answertype, iters):
        ''' uses logistic regression on trainings data'''
        # regression
        regression = LogisticRegression(max_iter=iters)
        #fit model
        regression = regression.fit(matrix, answertype)
        return regression

    def run(self):
        ''' default run '''
        train_path = "projekt/data/dataframes/train.csv"
        self.build_bag_of_words(train_path)
        matrix, answertype = self.read_train_or_test_data(train_path)
        self.regression = self.logistic_aggression(matrix, answertype, 50000)
        self.pickeln("projekt/data/pickel/regression_questions.sav", self.regression)
        self.pickeln("projekt/data/pickel/vocab.sav",self.vocab)

    def testing_aggression(self):
        ''' for testing '''
        test_path = "projekt/data/dataframes/test.csv"
        matrix, answertype = self.read_train_or_test_data(test_path)
        score = self.regression.score(matrix, answertype)
        print(score)


if __name__ == '__main__':
    classifier = Question_classifier_logistic()
    classifier.run()
    classifier.testing_aggression()