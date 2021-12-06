import pandas as pd
from sklearn import preprocessing
import numpy as np

def prepare_german_dataset(filename, path_data):
    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',')   
    
    return df

def prepare_compas_dataset(filename, path_data):
    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'
    df['class'] = df['decile_score'].apply(get_class)

    #Race discretization
    race = df['race']
    label_encoder = preprocessing.LabelEncoder()
    race_disc = label_encoder.fit_transform(race)
    df['race_d'] = race_disc
    del df['race']
    df['race'] = df['race_d']
    del df['race_d']

    #Sex discretization
    sex = df['sex']
    sex_disc = label_encoder.fit_transform(sex)
    df['sex_d'] = sex_disc
    del df['sex']
    df['sex'] = df['sex_d']
    del df['sex_d']

    #c_charge_degree discretization
    c_charge_degree = df['c_charge_degree']
    c_charge_degree_disc = label_encoder.fit_transform(c_charge_degree)
    df['c_charge_degree_d'] = c_charge_degree_disc
    del df['c_charge_degree']
    df['c_charge_degree'] = df['c_charge_degree_d']
    del df['c_charge_degree_d']

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']
    
    return df