import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
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

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), KNeighborsRegressor(), GradientBoostingRegressor()]

def extract_numbers_from_filename(filepath):
    filename = os.path.basename(filepath)
    match = re.match(r'(\d+)_(\d+)_data\.csv', filename)
    if match:
        icr_num, inv_num = match.groups()
        logging.debug(f"Extracted ICR number: {icr_num}, Inverter number: {inv_num} from filename: {filename}")
        return icr_num, inv_num
    else:
        logging.warning(f"Failed to extract numbers from filename: {filename}")
        return None, None

def evaluate_model(models, i, X_train, y_train, X_test, y_test, wantdf=False, wantprint=False):
    model = models[i]
    logging.debug(f"Training model: {type(model).__name__}")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    logging.debug(f"Model {type(model).__name__} prediction completed.")
    
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
    final['Error%'] = final['Error'] * 100 / final['Actualdcpower']
    
    logging.debug(f"Model {type(model).__name__} evaluation completed. Mean Error: {final['Error'].mean()}")
    
    if wantprint:    
        logging.info(f"Final DataFrame Description: {final.describe()}")
        print(final.describe())
        print("-----------------------------------------")
        print('\n')
        print('\n')
    if wantdf:
        return final

def application(i=0, X_train=None, y_train=None, X_test=None, y_test=None, wantprint=False):
    models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), KNeighborsRegressor(), GradientBoostingRegressor()]
    finaldf = evaluate_model(models, i, X_train, y_train, X_test, y_test, wantdf=True, wantprint=wantprint)
    
    model = models[i]
    model_name = type(model).__name__
    pickle_file = f"{icr_num}_{inv_num}_{model_name}_{i}.pickle"
    
    with open(pickle_file, "wb") as f:
        pickle.dump(model, f)
    
    logging.debug(f"Model {model_name} has been saved to {pickle_file}.")
    
    return 100 - ((finaldf['Error'].mean() / finaldf['Actualdcpower'].mean()) * 100)

def find_max_3_values_and_indices(lst):
    indexed_list = [(value, index) for index, value in enumerate(lst)]
    sorted_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)
    top_3 = sorted_list[:3]
    values = [item[0] for item in top_3]
    indices = [item[1] for item in top_3]
    
    logging.debug(f"Top 3 values: {values} with indices: {indices}")
    
    return values, indices

def ensembler_pred(indices, weights, model_names):
    model_list = []

    for i, weight in zip(indices, weights):
        with open(f'{icr_num}_{inv_num}_{model_names[i]}_{i}.pickle', 'rb') as f:
            model = pickle.load(f)
        model_list.append((model_names[i], model))

    ensemble_model = VotingRegressor(estimators=model_list, weights=weights)
    logging.debug("Ensemble model has been successfully created.")
    
    with open(f'{icr_num}_{inv_num}_final_model.pickle', 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    logging.debug(f"The final ensemble model has been saved to {icr_num}_{inv_num}_final_model.pickle.")
def train_ensemble(filepath):
    icr_num, inv_num = extract_numbers_from_filename(filepath)
    df = pd.read_csv(filepath)
    logging.info(f"Started processing ICR Number: {icr_num}, Inverter Number: {inv_num}")

    df_date = pd.read_csv(r'C:\Users\rachitgandhi2\OneDrive - KPMG\Desktop\Solar_Azure_Deploy\Final_Train\dates\8_2_data_old.csv')
    df['datetime'] = df_date['timestamp']
    df.set_index('datetime', drop=True, inplace=True)
    df.rename(columns={'ICR1.WMS_PRG.WMS.GLOBAL_TILT_IRRADIATION_Wm2': 'irradiance'}, inplace=True)
    
    df['sin_time'] = np.sin(2 * np.pi * pd.to_datetime(df.index).dayofyear / 365)
    df['cos_time'] = np.cos(2 * np.pi * pd.to_datetime(df.index).dayofyear / 365)
    
    df['deferred_10min'] = df['irradiance'].shift(10).rolling(window=10).mean()
    df['deferred_15min'] = df['irradiance'].shift(15).rolling(window=15).mean()
    df['deferred_30min'] = df['irradiance'].shift(30).rolling(window=30).mean()
    df['deferred_60min'] = df['irradiance'].shift(60).rolling(window=60).mean()
    df.dropna(inplace=True)
    X = df[['irradiance', 'deferred_10min', 'deferred_15min', 'deferred_30min', 'deferred_60min', 'sin_time', 'cos_time']]
    y = df[[f'icr{icr_num}_inv1_dcpower']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
    logging.debug("Training and testing data split completed.")
    
    raw_weights = [application(i, X_train, y_train, X_test, y_test) for i in range(5)]
    
    values, indices = find_max_3_values_and_indices(raw_weights)
    weights = [value / sum(values) for value in values]
    
    ensembler_pred(indices, weights, [type(model).__name__ for model in models])
    logging.info(f"Finished processing ICR Number: {icr_num}, Inverter Number: {inv_num}")


# Define the directory where the CSV files are located
directory = r'C:\Users\rachitgandhi2\OneDrive - KPMG\Desktop\Solar_Azure_Deploy\Final_Train'

# Define the pattern of the CSV files
pattern = '*_*_data.csv'

# Get the paths of all CSV files of the specified format in the directory
csv_files = glob.glob(os.path.join(directory, pattern))

# Print the paths of the CSV files
for csv_file in csv_files:
    icr_num, inv_num = extract_numbers_from_filename(csv_file)
    train_ensemble(csv_file)
