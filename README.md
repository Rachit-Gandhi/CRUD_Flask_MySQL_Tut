"""!pip install pandas
!pip install numpy
!pip install scikit-learn
!pip install seaborn
!pip install xgboost
!pip install statsmodels
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from itertools import product
import warnings
import re
import os
import glob
import pickle

def extract_numbers_from_filename(filepath):
    filename = os.path.basename(filepath)
    match = re.match(r'(\d+)_(\d+)_data\.csv', filename)
    if match:
        icr_num, inv_num = match.groups()
        return icr_num, inv_num
    else:
        return None, None

def evaluate_model(models, i, wantdf=False, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,wantprint = False):
    model = models[i]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Name of model:" + str(model))
    # If the model returns multiple values, take the first one
    if len(pred.shape) > 1:
        pred = [i[0] for i in pred]
    
    final = pd.DataFrame({'Predicteddcpower': pred}, index=y_test.index)
    final['irradiance'] = X_test['irradiance']
    final['deferred_10min'] = X_test['deferred_10min']
    final['deferred_15min'] = X_test['deferred_15min']
    final['deferred_30min'] = X_test['deferred_30min']
    final['deferred_60min'] = X_test['deferred_60min']
    final['sin_time'] = X_test['sin_time']
    final['cos_time'] = X_test['cos_time']
    final['Actualdcpower'] = y_test[f'icr{icr_num}_inv1_dcpower']
    final['Error'] = abs(final['Predicteddcpower'] - final['Actualdcpower'])
    final['Error%'] = final['Error']*100/final['Actualdcpower']
    #final['datetime'] = X_test.index
    
    if wantprint:    
        print(final.describe())
        print("-----------------------------------------")
        print('\n')
        print('\n')
    if wantdf:
        return final



def application(i=0, wantprint=False):
    models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), KNeighborsRegressor(), GradientBoostingRegressor()]
    model = models[i]
    model.fit(X_train, y_train)
    finaldf = evaluate_model(models, i, wantdf=True, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, wantprint=True)
    
    # Save the trained model to a pickle file
    model_name = type(model).__name__
    pickle_file = f"{icr_num}_{inv_num}_{model_name}_{i}.pickle"
    with open(pickle_file, "wb") as f:
        pickle.dump(model, f)
    
    return (100-((finaldf['Error'].mean()/finaldf['Actualdcpower'].mean())*100))
def find_max_3_values_and_indices(lst):
    # Create a list of tuples (value, index)
    indexed_list = [(value, index) for index, value in enumerate(lst)]
    
    # Sort the list based on values in descending order
    sorted_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)
    
    # Get the top 3 values and their indices
    top_3 = sorted_list[:3]
    
    # Separate the values and indices
    values = [item[0] for item in top_3]
    indices = [item[1] for item in top_3]
    
    return values, indices
import pickle
import numpy as np
from sklearn.ensemble import VotingRegressor

# Define the weights, indices, and models

models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), KNeighborsRegressor(), GradientBoostingRegressor()]
model_names = []
for model in models:
    model_names.append(type(model).__name__)

def ensembler_pred(indices,weights,model_names):


    # Initialize an empty list to store the models
    model_list = []

    # Iterate over each index and weight
    for i, weight in zip(indices, weights):
        # Unpickle the model
        with open(f'{icr_num}_{inv_num}_{model_names[i]}_{i}.pickle', 'rb') as f:
            model = pickle.load(f)
        
        # Add the model to the model list
        model_list.append((model_names[i], model))

    # Create an ensemble model
    ensemble_model = VotingRegressor(estimators=model_list, weights=weights)

    # Print a success message
    print("The ensemble model has been successfully created.")
    with open(f'{icr_num}_{inv_num}_final_model.pickle', 'wb') as f:
        pickle.dump(ensemble_model, f)
    print(f"The final model has been successfully pickled to {icr_num}_{inv_num}_final_model.pickle.")

warnings.filterwarnings('ignore')
def train_ensemble(filepath):
    df = pd.read_csv(filepath)
    print(f"ICR Number: {icr_num}, Inverter Number: {inv_num} started processing")

    icr_num, inv_num = extract_numbers_from_filename(filepath)
    #df.set_index(pd.to_datetime(df.index), drop=True, inplace=True)
    df_date = pd.read_csv('dates\8_2_data_old.csv')
    df['datetime'] = df_date['timestamp']
    df.set_index('datetime', drop=True, inplace=True)
    df.rename(columns={'ICR1.WMS_PRG.WMS.GLOBAL_TILT_IRRADIATION_Wm2' : 'irradiance'}, inplace=True)
    df['sin_time'] = np.sin(2 * np.pi * pd.to_datetime(df.index).dayofyear / 365)

    # Create a new column with the cosine representation of the datetime index
    df['cos_time'] = np.cos(2 * np.pi * pd.to_datetime(df.index).dayofyear / 365)

    # Create deferred variable columns for 10, 15, 30, 60 mins

    df['deferred_10min'] = df['irradiance'].shift(10).rolling(window=10).mean()
    df['deferred_15min'] = df['irradiance'].shift(15).rolling(window=15).mean()
    df['deferred_30min'] = df['irradiance'].shift(30).rolling(window=30).mean()
    df['deferred_60min'] = df['irradiance'].shift(60).rolling(window=60).mean()
    X = df[['irradiance','deferred_10min','deferred_15min','deferred_30min','deferred_60min','sin_time','cos_time']]
    y = df[[f'icr{icr_num}_inv1_dcpower']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
    raw_weights = []
    for i in range(0,4):
        raw_weights.append(application(i))
    values, indices = find_max_3_values_and_indices(raw_weights)
    sum = 0
    for value in values:
        sum += value
    weights = []
    for value in values:
        weights.append(value/sum)
    ensembler_pred(indices,weights,model_names)
train_ensemble('8_1_data.csv')
