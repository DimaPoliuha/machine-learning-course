# Eclat

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(7501):
    transactions.append(list(str(dataset.values[i,j]) for j in range(20)))
    
# Training Eclat on the dataset
from apyori import apriori
#min_support = 3 * 7 / 7500 # The product that people buy 3 days per day
rules = apriori(transactions, min_support = 0.003, min_confidence = 0, min_lift = 0, min_length=2)

# Visualising the Results
results = list(rules)
clean_results = []
for i in range(0, len(results)):
    result_dict = dict()
    result_dict["RULE"] = list(results[i][0])
    result_dict["SUPPORT"] = results[i][1]
    result_dict["CONFIDENCE"] = results[i][2][0][2]
    result_dict["LIFT"] = results[i][2][0][3]
    clean_results.append(result_dict)