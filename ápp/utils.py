import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_obj(name ):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)

# load trained model and standardizer
PATH = "final_model.pkl"

loaded_model = pickle.load(open(PATH, 'rb'))

standardizer = load_obj('standardizer')

# define transform function
def transform_data(raw_data):
    raw_data = raw_data.fillna(0)
    raw_data['Age'] = standardizer.transform(raw_data['Age'].values.reshape(-1, 1))
    return raw_data

# define prediction function
def get_prediction(transformed_data):
    pred = loaded_model.predict_proba(transformed_data)
    return pred
