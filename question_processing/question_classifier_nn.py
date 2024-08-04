import random
import torch
import torch.nn as nn
from nltk.stem import PorterStemmer
import re
import nltk
import numpy as np
import torch.optim as optim
import pandas as pd

stem = nltk.stem.PorterStemmer()
SAVE_MODEL_PATH = 'blatt8/model/test1.pth'

class Question_classifier_nn():

    def __init__(self, Net, train_path, test_path,hidden = 500, epochs=30, lr=0.01):
        self.Net = Net
        self.epochs = epochs
        self.hidden = hidden
        self.lr = lr
        self.train_path = train_path
        self.test_path = test_path

    def run_train_and_test(self):
        '''runs the training and testing'''

        self.create_train_test_data()
        self.create_new_mlp()
        self.run_training()
        self.testing()


    def create_train_test_data(self):
        '''creates the train and test data'''

        print("Creatting train and test data")
        self.build_bag_of_words(self.train_path)
        self.train_matrix, self.train_answers = self.read_train_or_test_data(self.train_path)
        self.test_matrix, self.test_answers = self.read_train_or_test_data(self.test_path)
    
    def create_new_mlp(self):
        '''creates a new MLP'''

        print("Creation of new MLP")
        self.mlp = Net(len(self.bag_of_words), self.hidden, len(self.train_answers[0]))

    def save_model(self, save_path):
        '''saves the model'''

        print("Saving model to: " + save_path)
        torch.save(self.mlp.state_dict(), save_path)

    def load_model(self, load_path):
        '''loads the model'''
        self.create_train_test_data()
        self.create_new_mlp()
        print("Loading model from: " + load_path)
        self.mlp.load_state_dict(torch.load(load_path))

    def build_bag_of_words(self, path):
        '''builds the bag of words'''

        tokens = set()
        
        with open(path) as file:
            for line in file:
                line = line.split(";")

                linearr = nltk.word_tokenize(line[1])
                linearr = [ele.lower() for ele in linearr if not re.compile(r'[^a-zA-Z]+').match(ele)] # für jedes Wort checken, dass es kein Sonderzeichen/Zahl, dann in lowerform in die Liste hinzufügen
                
                for ele in linearr:
                    tokens.add(stem.stem(ele)) 

        self.bag_of_words = list(tokens)

    

    def stemm_question(self, question):
        '''returns a stemmed question'''
        linearr = nltk.word_tokenize(question)
        linearr = [ele.lower() for ele in linearr if not re.compile(r'[^a-zA-Z]+').match(ele)] # für jedes Wort checken, dass es kein Sonderzeichen/Zahl, dann in lowerform in die Liste hinzufügen
        for i in range(len(linearr)):
            linearr[i] = stem.stem(linearr[i])
        return linearr

    def create_matrix(self, questions):
        '''creating a matrix with the quesions and the vocab'''
        tokens = self.bag_of_words
        matrix = np.zeros(shape=(len(questions),len(tokens)))

        for i in range(len(questions)):
            for j in range(len(tokens)):
                if tokens[j] in questions[i]:
                    matrix[i][j] = 1    
        return matrix

    def read_train_or_test_data(self, path):
        '''reads training or test data and returns anwsertyps and matrix'''
        df = pd.read_csv(path, sep = ";")
        df["question"] = df["question"].apply(self.stemm_question)
        questions = np.array(df["question"])
        answertype = np.array(df["answer_type"])
        matrix = self.create_matrix(questions)
        self.answertypes, answertype = self.convert_anwsertype_to_number(answertype)
        return matrix, answertype

    def convert_anwsertype_to_number(self, answertype):
        '''converts anwsertype to number'''
        answertype = list(answertype)
        answertypes = list(set(answertype))
        matrix = np.zeros(shape=(len(answertype), len(answertypes)))
        count = 0
        for ele in answertype:
            matrix[count][answertypes.index(ele)] = 1
            count += 1
        return answertypes, matrix


    def minibatch(self, X, t, batch_size):
        '''returns a minibatch'''
        n = len(X)
        all_index = list(range(n))
        some = np.array( random.sample(all_index, batch_size))
        X_Batch = list()
        t_Batch = list()
        for i in some:
            X_Batch.append(X[i])
            t_Batch.append(t[i])


        X_ = torch.tensor(np.array(X_Batch).astype(np.float32))
        t_ = torch.tensor(np.array(t_Batch).astype(np.float32))
        return X_,t_

    def to_shuffled_tensor(self, X, t):
        '''returns a shuffled tensor from data'''
        n = len(X)
        all_index = list(range(n))
        some = np.array( random.sample(all_index, len(X)))
        X_Batch = list()
        t_Batch = list()
        for i in some:
            X_Batch.append(X[i])
            t_Batch.append(t[i])


        X_ = torch.tensor(np.array(X_Batch).astype(np.float32))
        t_ = torch.tensor(np.array(t_Batch).astype(np.float32))
        return X_,t_

    def run_training(self):
        '''runs the training'''
        optimizer = optim.Adam(self.mlp.parameters(), self.lr) 
        criterion = nn.CrossEntropyLoss()
        self.mlp.train()
        count = 0
        for i in range(self.epochs):
            x_, t_ = self.to_shuffled_tensor(self.train_matrix, self.train_answers)
            # set gradients to zero
            optimizer.zero_grad()

            #forward pass
            y_ = self.mlp(x_)
            loss = criterion(y_, t_)
            loss.backward()
            optimizer.step()

            #if i % 2 == 0:
            print(f'Iteration {i} : Loss {loss.item()}')
            
            count += 1
        
        #torch.save(mlp.state_dict(), SAVE_MODEL_PATH)

    def testing(self):
        '''testing the model'''
        self.mlp.eval()
        X = self.test_matrix
        t = self.test_answers
        total = len(X)
        correct = 0

        with torch.no_grad(): # no need to track gradients
            output = self.mlp(torch.tensor(np.array(X).astype(np.float32)))
            #print(output)

            for i in range(len(output)):
                correct += list(t[i]).index(max(t[i])) == list(output[i]).index(max(output[i]))
        
        print(f'Accuracy: {correct/total}')
    
    def predict_question(self, question:str):
        '''predicts a given question'''

        X = self.create_matrix([question])
        self.mlp.eval()

        with torch.no_grad(): # no need to track gradients
            output = self.mlp(torch.tensor(np.array(X).astype(np.float32)))
            ele = output[0].argmax().item()
            return self.answertypes[ele]



class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), output_size)
        self.tanh = nn.Tanh()
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x


if '__main__' == __name__:
    
    train_path = "projekt/data/dataframes/train.csv"
    test_path = "projekt/data/dataframes/test.csv"
    eval_path = "projekt/data/dataframes/eval.csv"  
    modelPath = 'projekt/data/mlp_model/test.pth'
  
    qc = Question_classifier_nn(Net2, train_path, test_path, 500, 100, 0.002)
    qc.create_train_test_data()
    qc.create_new_mlp()
    qc.run_training()
    qc.testing()
    qc.save_model(modelPath)
    answer = qc.predict_question("Where was James Watt born?")
    print(answer)