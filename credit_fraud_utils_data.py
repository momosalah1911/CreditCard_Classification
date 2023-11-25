import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler


def load_dataset(url=r"D:\Data Science\course_ML_mostafa_saad\projects\2 Credit Card Fraud Detection\data\split\train.csv"):
    df = pd.read_csv(rf"{url}")
    return df


def split_to_data_target(df):
    target = df['Class']
    data = df.drop(['Class'], axis=1)
    return target, data


def normalized_data(train_x, choice=0):
    if choice == 0:
        scale = MinMaxScaler()
    else:
        scale = StandardScaler()
    train_x = scale.fit_transform(train_x)
    return train_x, scale


def check_for_nan_and_doublicate(df):
    if df.isna().sum().any() == True:
        df = df.dropna()
    if df.duplicated().sum().any() == True:
        df = df.drop_duplicates()
    return df


def transfor_log1p(df, features):
    df[features] = np.log1p(df[features])
    return df


def under_sampling_major(df, X, y, feature, factor):
    Class_values = df[feature].value_counts().sort_values()
    manority = Class_values.keys()[0]
    manority_count = Class_values[manority]
    majority = Class_values.keys()[1]
    majority_count = Class_values[majority]

    rus = RandomUnderSampler(
        sampling_strategy={majority: factor*manority_count})
    X, y = rus.fit_resample(X, y)
    return X, y


def over_sampling_monority(df, X, y, feature, factor):
    Class_values = df[feature].value_counts().sort_values()
    manority = Class_values.keys()[0]
    manority_count = Class_values[manority]
    majority = Class_values.keys()[1]
    majority_count = Class_values[majority]
    rus = SMOTE(
        sampling_strategy={manority: int(majority_count/factor)})
    X, y = rus.fit_resample(X, y)
    return X, y


def under_and_over_sampling(df, X, y, feature):
    Class_values = df[feature].value_counts().sort_values()
    manority = Class_values.keys()[0]
    manority_count = Class_values[manority]
    majority = Class_values.keys()[1]
    majority_count = Class_values[majority]
    rus = SMOTEENN()
    X, y = rus.fit_resample(X, y)
    return X, y


def add_column_for_huage_amount(df, q9):
    df['huge_amount'] = df['Amount'] >= q9
    return df


def add_column_for_small_amount(df, q1):
    df['small_amount'] = df['Amount'] <= q1
    return df
