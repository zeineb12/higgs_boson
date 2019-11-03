# Machine Learning Project 1 - Higgs Boson Dataset
Code in Python using Numpy only. No external libraries were allowed except for visualisation purposes.

# Team Members

Zeineb Sahnoun: zeineb.sahnoun@epfl.ch

Sarah Antille: sarah.antille@epfl.ch

Lilia Ellouz: lilia.ellouz@epfl.ch

### Aim :
The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles have mass.
We are given a vector of features representing the decay signature of a collision event, and we want to predict whether this event was signal (a Higgs boson) or background (something else). 
To do this, we use different binary classification techniques and compare the results.

### Dataset :
Dataset available under this link: https://www.kaggle.com/c/higgs-boson/data 

It cannot be uploaded because of its size.

### Result:
We got a score of 81% in crowdAI.

### Files
- `implementations.py` : contains the implementation of all the machine-learning methods seen in class
- `helpers.py` : contains different methods for preprocessing and feature engineering we created and used for our predictions
- `proj1_helpers.py` : contains functions to load the dataset and create the csv file of our final predicitons
- `output_final.csv` : our prediction file that got us our score in the Kaggle competition
- `project1.ipynb` : jupyter notebook that helps to visualize the data and take you step by step through our methodology
- `run.py` : script that produces a .csv file for our predicitions

### Reproducibility
The script `run.py` produces exactly the same predictions (as a csv file) which we used in our best submission to the competition system.
