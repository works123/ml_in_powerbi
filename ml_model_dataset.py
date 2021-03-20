import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import pickle


acc_ix, hpower_ix, cyl_ix = 4, 2, 0

##custom class inheriting the BaseEstimator and TransformerMixin
class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True):
        self.acc_on_power = acc_on_power  # new optional variable
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix] # required new variable
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl] # returns a 2D array
        
        return np.c_[X, acc_on_cyl]


##handling missing values
def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe 
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
        
    '''
    numerics = ['float64', 'int64']

    num_attrs = data.select_dtypes(include=numerics)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
        ])
    return num_attrs, num_pipeline

##preprocess the Origin column in data
def preprocess_origin_cols(df):
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})    
    return df


def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    nuerical and categorical data.
    
    Argument:
        data: original dataframe 
    Returns:
        prepared_data: transformed data, ready to use
    '''
    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
        ])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data

def predict_mpg(config, model):
    
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    # data transformation pipeline
    preproc_df = preprocess_origin_cols(df)
    prepared_df = pipeline_transformer(preproc_df)

    #print(len(prepared_df[0]))
    y_pred = model.predict(prepared_df)
    return y_pred

def load_model(model_file_name_path):
    ml_model = None

    with open(model_file_name_path, 'rb') as f_in:
        ml_model = pickle.load(f_in)
    return ml_model


### page entry level 
# run model and create new dataset with predicted value 
def retrieve_clean_data(raw_data_file_path, model_file_name_path):
   
    # load raw data
    cols = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin', 'Car']
    df = pd.read_csv(raw_data_file_path, header=None, sep='\s+', names=cols, na_values='?')

    # make a copy of raw data
    df_original = df.copy()

    # extract columns required by model in dataset
    df = df.loc[:,['Cylinders', 'Displacement', 'Horsepower', 'Weight',
       'Acceleration', 'Model Year', 'Origin']]
    
    # convert the model file to a model object 
    ml_model = load_model(model_file_name_path)

    #get perdiction 
    model_result = predict_mpg(df, ml_model)


    #final result
    df_original['mpg_predict'] = model_result

    return df_original

if __name__=='__main__':
    # location of raw data and model file 
     data_file = "C:/Users/ADEBAYO ADERIBIGBE/OneDrive/Documents/Projects/ML_In_Production/ml_in_powerbi/data/car_data"
     model_file = "C:/Users/ADEBAYO ADERIBIGBE/OneDrive\Documents/Projects/ML_In_Production/ml_in_powerbi/model.pkl"
    
     # retrieve updated dataset with model prediction
     df_result = retrieve_clean_data(data_file, model_file)

print(df_result.head())
