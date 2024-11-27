import pandas as pd
import numpy as np
from graph_tool.all import *
from numpy.random import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error

from arc_delay_prediction_utils import *


def generate_dataset(IR_path):
    # Generate LPG and properties dataframe
    g, master_df_dict = generate_LPG_from_tables(IR_path)
    
    # Extract required dataframes from master dict
    pin_df = master_df_dict["pin_df"]
    pin_pin_df = master_df_dict["pin_pin_df"]
    cell_df = master_df_dict["cell_df"]
    
    # Get only pin arcs inside cells
    cell_pin_arcs_df = filter_pin_arcs(pin_pin_df,"cell")
    
    # Calculate load capacitance
    output_pins = cell_pin_arcs_df['tar_id'].unique()
    cell_pin_arcs_df = calculate_load_cap(output_pins, cell_pin_arcs_df)
    
    # Merge input tran and cell type to pins
    cell_pin_arcs_df = merge_tran_cell(cell_pin_arcs_df, pin_df, cell_df)
 
    return cell_pin_arcs_df

def generate_ML_data(cell_pin_arcs_df):
    # ML data extraction
    # Features: input tran, oad cap, cell type
    # Label: arc delay

    data = cell_pin_arcs_df.loc[:,['pin_tran','output_cap','cell_type_coded','arc_delay']]
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Splitting into train and test datasets
    X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizing the feature columns
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    return X_train_normalized, X_test_normalized, y_train, Y_test

def train_rf_model(X_train, Y_train):
    # Training the RF model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test):
    # Making predictions
    Y_predict = model.predict(X_test)
    
    mse = mean_squared_error(Y_test, Y_predict)
    mae = mean_absolute_error(Y_test,Y_predict)
    rmse = np.sqrt(mse)
    max_err = max_error(Y_test, Y_predict)
    
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    
    Y_predict = Y_predict.reshape(-1)
    errors = Y_predict - Y_test
    abs_errors = abs(errors)
    error_percentages = (abs_errors / Y_test) * 100
    mean_percent = sum(error_percentages) / len(error_percentages)
    print("Max Error : ", max(abs_errors))
    print("Maximum Error %:", max(error_percentages))
    print("Mean Error %:", mean_percent)

if __name__ == "__main__":

     # Set the Circuitops path
    cops_path = "/home/vgopal18/Circuitops/CircuitOps/IRs/"
    design_name = "gcd"
    platform = "nangate45"

    IR_path = f"{cops_path}/{platform}/{design_name}/"
    
    cell_pin_arcs_df = generate_dataset(IR_path)

    X_train, X_test, Y_train, Y_test = generate_ML_data(cell_pin_arcs_df)

    model = train_rf_model(X_train, Y_train)

    evaluate_model(model, X_test, Y_test)
