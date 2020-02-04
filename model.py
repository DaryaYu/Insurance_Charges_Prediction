
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv('InsuranceCost.csv')

df.drop_duplicates(inplace=True)

x = df.iloc[:, :5]
y = df.iloc[:, -1]

regressor = LinearRegression()
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

