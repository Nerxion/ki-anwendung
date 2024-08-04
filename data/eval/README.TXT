This code creates an HTML scoreboard for question answering.



0. AUTHOR
---------------------
- Adrian Ulges
  (adrian.ulges@hs-rm.de)


1. QUICK INSTRUCTIONS
---------------------
Run the following command:

> python3 eval_main.py

You will find a visualization with two dummy teams for two dummy questions
in 'scoreboard.html'. Simply open the scoreboard in the browser, e.g.

> firefox scoreboard.html



2. RUN WITH YOUR OWN DATA.
---------------------
To run the evaluation with your own results, choose a folder where
to place the evaluation data. Set this folder in eval_main, e.g.:

> eval_main.PATH = './team_results'

In this folder, place a ground truth file with questions+answers. This 
should come from the course instructor, and be a ;-separated csv file.
Each line should contain:
    1. the name of the team that contributed the question
       (not really releavnt for the evaluation).
    2. the question itself.
    3. the ground truth answer.
    4. the answer type (e.g., HUM:ind).
    5. the ID of the wikipedia page containing the answer
       (from the wiki base).
    6. the sentence containing the answer.

Also, for each team to be evaluated (e.g., "NaturalIntelligence"),
place an answer csv file with the team's name in the folder, e.g.:

> ./teamresults/NaturalIntelligence.csv

This csv should have exactly as many lines as the ground truth file.
Each line corresponds the answers to the corresponding question in the
ground truth file, ;-separated, starting with the supposedly best one.

Finally, add your team to the TEAMS dictionary in eval_main, e.g.:

> eval_main.TEAMS = { "NaturalIntelligence",
                      "AI-Dojo",
		      ...
		    }

You should now be able to run the script (see 1.) and generate the
scoreboard.



3. METRIC COMPUTATION
---------------------
The code in eval.py computes the evaluation metric by which we determine
the contest winner. Have a detailed look, especially at eval.eval().
