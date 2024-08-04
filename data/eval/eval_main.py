#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import os
import csv
import random
import itertools
import nltk
import sys
import numpy as np
import operator
from eval import eval, K
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

"""
'ANWENDUNGEN DER KI': QA EVALUATION
---------------------------------
This code is used for final evaluation
and generates an HTML scoreboard.

Check out README.TXT!
"""


# enter your teams here!
TEAMS = {
    'miler'
}


# enter a folder where you placed each team's CSV here.
# For each team, there should be a file
# PATH + / + <TEAMNAME>.csv.
PATH = './projekt/data/eval/our_results'


# each team gets a color...
CMAP = plt.get_cmap('tab20')
COLORS = dict([(team,CMAP(i)) for i,team in enumerate(sorted(TEAMS))])



def read_questionfile(f):
    """
    read a list of questions from a csv file (see 'questions.csv').
    Each line contains (semicolon-separated):
    - the name of the team that contributed the question.
    - the question itself.
    - the ground truth answer.
    - the question type (e.g., HUM:ind).
    - the ID of the wikipedia page (in the wiki base).
    - the sentence containing the answer.
    """
    result = []

    with open(f, newline='', encoding="utf-8") as csvfile:
        csvlines = csv.reader(csvfile, delimiter=';')
        for team, question, answer, answertype, pageid, passage in csvlines:
            result.append({'team': team,
                           'question': question,
                           'answer': answer,
                           'answertype': answertype,
                           'pageid': int(pageid),
                           'passage': passage})
    return result



def read_answerfile(f):
    """
    read answers to a list of questions.
    Each line contains all answers to a question, semicolon-separated,
    ranked from left to right. (see 'answers.csv')
    """
    result = []
    with open(f, newline='', encoding="utf-8") as csvfile:
        csvlines = csv.reader(csvfile, delimiter=';')
        for line in csvlines:
            answers = line[:K]
            result.append(answers)
    return result



def HEADLINE(s):
    return '<p><table border=0 width="100%%"><tr>'\
        '<td align=left bgcolor="#ff9600"><font color="#ffffff" size="+3">'\
        '&nbsp;&nbsp;&nbsp;%s</font></td></tr></table><p>' %s






if __name__ == "__main__":

    questions = read_questionfile(PATH + os.sep + 'test.csv')
    gt_answers = [q['answer'] for q in questions]
    N = len(questions)

    teams = sorted(TEAMS)
    
    MRRS = {}
    ANSWERS = {}
    ALL_ANSWERS = {}
    RRS = {}
    CUMRRS = {}

    # evaluate one team.
    for team in teams:

        answerfile = PATH + os.sep + team + '.csv'
        
        # read input file
        print('Reading File...', answerfile)
        system_answers = read_answerfile(answerfile)

        # run evaluation
        print('EVALUATING TEAM', team)
        mrr, best_rrs, best_answers = eval(gt_answers, system_answers)

        MRRS[team]        = mrr
        ANSWERS[team]     = best_answers
        RRS[team]         = best_rrs
        ALL_ANSWERS[team] = system_answers        
        CUMRRS[team]      = np.cumsum(RRS[team])
                

    # read HTML template for scoreboard
    with open("./projekt/data/eval/template.html") as input:
        html = input.read()


    #########################################################
    # 1. High Scores: ranked list of teams with their scores
    #########################################################

    html += HEADLINE('High Scores')
    html += '<table class="hs" bgcolor="#DDDDDD">'
    html += '<tr>'
    html += '<td><h2>rank</font></h2></td>'
    html += '<td><h2>team</font></h2></td>'
    html += '<td><h2>score(percent)</font></h2></td>'
    
    for i,(team,score) in enumerate(sorted(MRRS.items(),
                                           key=operator.itemgetter(1),
                                           reverse=True)):
        html += '<tr>'
        html += '<td><h2><font color="%s">%d.</font></h2></td>' %(to_hex(COLORS[team]),i+1)
        html += '<td><h2><font color="%s">%s</font></h2></td>' %(to_hex(COLORS[team]),team)
        html += '<td><h2><font color="%s">%.1f</font></h2></td>' %(to_hex(COLORS[team]),score*100)

    html += '</table>'
    

    #########################################################
    # 2. Plot one curve per team (running MRR over questions)
    #########################################################

    html += HEADLINE('Accumulated Score plotted over Questions')

    import matplotlib.pyplot as plt, mpld3
    import matplotlib
    matplotlib.rcParams.update({'font.size': 32,
                                'figure.autolayout': True})
    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'), figsize=(18,13))
    ax.grid(True, alpha=0.3)
    plt.xlabel('questions')
    plt.ylabel('accumulated score')

    for team in teams:
        res = ax.plot(range(1,N+1), CUMRRS[team], 'ks-', mec='w', color=COLORS[team], mew=1)
        labels = ['%s' %team for i in range(N)]
        tooltip = mpld3.plugins.PointLabelTooltip(res[0], labels=labels)
        mpld3.plugins.connect(fig, tooltip)

    html +=  mpld3.fig_to_html(fig)

    
    #########################################################
    # 3. Details: For each question, show the best answers.
    #########################################################

    html += HEADLINE('Details per Questions')

    for i,q in enumerate(questions):

        fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'), figsize=(5,3))

        print(i, '...')
        rrs = [RRS[team][i] for team in teams]
        answers = [ANSWERS[team][i] for team in teams]
        data = sorted(zip(teams,rrs,answers), key=operator.itemgetter(1), reverse=True)

        # create a barplot
        matplotlib.rcParams.update({'font.size': 9,
                                    'figure.autolayout': True})
        bestidx = [j for j in range(len(teams)) if rrs[j]==max(rrs)]
        #best = teams[random.choice(bestidx)]
        best_answer = ','.join(['<font color="%s">%s</font>' %(to_hex(COLORS[teams[b]]),teams[b]) for b in bestidx])
        html += u'<H2>&nbsp;&nbsp;&nbsp;{0}. {1}</H2><H3>&nbsp;&nbsp;&nbsp;answer: {3}. <H3>&nbsp;&nbsp;&nbsp;best guess: {4}</H3> '.format(i+1,
                                                                                                                                            q['question'],
                                                                                                                                            2,
                                                                                                                                            q['answer'],
                                                                                                                                            best_answer if max(rrs)>0 else '',
                                                                                                                                            5)
        xticks = [0.5+i for i in range(len(data))]
        boxes = ax.bar(xticks, [r for t,r,a in data],
                       #title=
                       align='center', width=0.9,alpha=0.7,
                       facecolor='red',
                       edgecolor='w')
        labels = [a for t,r,a in data]
        for i,b in enumerate(boxes):
            boxes[i].set_color(COLORS[data[i][0]])
        for i,b in enumerate(boxes.get_children()):
            tooltip = mpld3.plugins.LineLabelTooltip(b, label=labels[i])
            mpld3.plugins.connect(plt.gcf(), tooltip)

        html += mpld3.fig_to_html(fig)
        fig.clf()
        
    html += '</html>'


    # write stuff to output file.
    with open('scoreboard.html', 'w', encoding="utf-8") as stream:
        stream.write(html)
