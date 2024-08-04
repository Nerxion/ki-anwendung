#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import random
import itertools
import nltk
import sys
import numpy as np
import operator


'''
EVALUATING QUESTION ANSWERING
-----------------------------------

Given a list of ground truth answers (each belonging to a test question)
and lists of answers estimated by AI systems, this code computes a 
modified MRR metric: 

1. MATCH SCORE
--------------
For a given ground truth answer ("Harry Potter") and estimated answer
("Harry James Potter"), we first compute a MATCH SCORE (here, 2/3) indicating
how well the answers overlap in terms of tokens.

2. ANSWER QUALITY
-----------------
An answer's quality is computed by multiplying the match score with
the answers RECIPROCAL RANK (see lecture):

  QUALITY   =   RECIPROCAL_RANK  x  MATCH_SCORE

3. OVERALL QUALITY
-----------------
The overall quality of a system is computed by picking the HIGHEST-QUALITY
answer per test question, and then averaging this over all test questions.
'''


# We preprocess answers before comparing them,
# including a stemming ...
PORTER = nltk.PorterStemmer()

# ... and a stopword filtering.
def read_stopwords(path):
    with open(path) as stream:
        lines = stream.readlines()
    stopwords = [l.rstrip().lower() for l in lines]
    return stopwords

STOPWORDS = set(read_stopwords("./projekt/data/eval/stopwords.txt"))

# We only consider the top-K answers when computing
# our MRR evaluation measure.
K = 10






def answer_to_tokens(answer):
    """
    Given an answer string, tokenize it and
    preprocess it, and remove duplicates.

    returns a set of (preprocessed) tokens.
    """
    # split tokens: 'mind-blowing' -> 'mind', 'blowing'
    tokens = [t.split('-') for t in answer.split()]
    tokens = list(itertools.chain.from_iterable(tokens))

    # remove trailing or leading dots and apostrophs
    tokens = [t.strip('.') for t in tokens]
    tokens = [t.strip("'") for t in tokens]

    # stemming
    for i,t in enumerate(tokens):
        try:
            tokens[i] = PORTER.stem(t)
        except:
            pass

    # lowercasing
    tokens = [t.lower() for t in tokens]

    # stopword filtering
    tokens = [t for t in tokens if t not in STOPWORDS]

    return set(tokens)



def is_number(s):
    """
    returns True iff the given string is a number
    """
    try:
        _ = float(s)
        return True
    except:
        return False


    
def answer_to_number(answer):
    """
    We define a 'numeric' answer as an answer that consists
    mainly of a number:
    o 1991                 -> 1991
    o December 26, 1991    -> 1991
    o 13,000 years ago     -> 13000
    o 277,923.1 km2.       -> 277923.1

    This method returns a tuple (is_numeric, number), where
    o 'is_numeric' is True iff the answer appears to be 'numeric'.
    o 'number'     is the extracted number.
    """
    answer = answer.replace('$', '') # $470
    answer = answer.replace(',', '') # 13,000 years
    tokens = answer.split()
    numbers = [t for t in tokens if is_number(t)]

    if len(tokens) <=3 and len(numbers) >= 1:
        return True, numbers[-1]
    else:
        return False, numbers[-1] if len(numbers)>0 else None



def match_score_numeric(gt_answer, est_answer):
    """
    computes the match score for numeric answers.
    The result is either 1 (iff ground truth answer 
    and estimated answer contain the same number) or 0.
    """
    gt_isnumeric,gt_number = answer_to_number(gt_answer)
    est_isnumeric,est_number = answer_to_number(est_answer)
    if not gt_isnumeric:
        return 0.
    elif gt_number is not None and est_number is not None:
        return float(int(float(gt_number))==int(float(est_number)))
    else:
        return 0.


    
def match_score_strings(gt_answer, est_answer):
    """
    computes a match score between two answers based on term overlap.
    The result is in the range [0,1], and is higher the more
    terms match.
    """
    tokens1 = answer_to_tokens(gt_answer)
    tokens2 = answer_to_tokens(est_answer)
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return 0. if union==0 else intersection / union




def match_score(gt_answer, est_answer):
    """
    Given a ground truth answer (gt_answer) and an
    estimated  answer (est_answer), compute the
    overlap between the two (in percent). 

    Since we do not know if answers are numeric or not,
    just choose the better of the two scores.
    """
    return max(match_score_strings(gt_answer, est_answer),
               match_score_numeric(gt_answer, est_answer))




def eval(gt_answers, est_answer_lists):
        """
        This method measures how well a QA system's answers fit
        the correct ("ground tuth") answers to a list of n test questions.

        Given are
        1. a list of n ground truth answers (gt_answers). 
        2. est_answer_lists, which contains n lists of 
           estimated answers (ranked, starting with the best).

        The method computes a modified mean reciprocal rank (MRR):
        It includes a match score between ground truth and estimated answers:
        (a) for string-type answers, the match score is based on term overlap.
        (b) for numeric answers, the score is 1 iff both answers contain the same number
            (e.g., "Dec 1991" matches "1991").

        @param gt_answers: The list of n answers to test questions 
                           (each a string).
        @type gt_answers: list<string>

        @param est_answer_lists: A list of n elements. Each of those 
                                 contains a *list* of answers produced 
                                 by your QA system for the associated test question. 
                                 This list is assumed 
                                 to be ranked, with the best answer 
                                 appearing as first, the second-best answer
                                 appearing as second, etc. 
                                 Only the top K of these answers will be
                                 used for evaluation.
        @type est_answer_lists: list<list<string>>

        @returns: 1. the MRR score (our overall performance measure).
                  2. a list with each question's RR score.
                  3. each question's highest-scored answer. 
        @rtype: (float, np.array<float>, list<string>)
        """
        # consider only the top K answers
        est_answer_lists = [l[:K] for l in est_answer_lists] 

        all_scores = np.zeros(len(gt_answers))
        all_best_answers = [None] * len(gt_answers)

        for i,(gt_answer,est_answers) in enumerate(zip(gt_answers, est_answer_lists)):

            print('... comparing "%s" vs. "%s"' %(gt_answer, est_answers))

            if len(est_answers)==0:
                print('Warning: No answer for question no.', i, '(%s) !' %gt_answer)
                continue

            scores = np.zeros(len(est_answers))
            for j,answer in enumerate(est_answers):
                rrank = 1. / (j+1)
                match = match_score(gt_answer, answer)
                scores[j] = rrank * match
                
            best   = np.argmax(scores)
            all_scores[i] = scores[best]
            all_best_answers[i] = est_answers[best]

        return np.mean(all_scores), all_scores, all_best_answers



    
def small_test():
    gt_answers = ['John Wilkes Booth', '1947', 'xxx']    
    est_answer_lists = [ ['apple', 'cherry', 'J.W. Booth', 'John BOOTH'],
                         ['21 Jan 1948', '1948', 'Jan 1947', 'apple', 'lord Buckethead'],
                         ['']
                       ]
    mrr, rrs, best_answers =  eval(gt_answers, est_answer_lists) 
    print ("Overall MRR = ", mrr)
    for i,gt_answer in enumerate(gt_answers):
        print('Question 1: ground truth = %s, best answer = %s (score %.4f)'
              %(gt_answer, best_answers[i], rrs[i]))


        
if __name__ == "__main__":

    small_test()
    
