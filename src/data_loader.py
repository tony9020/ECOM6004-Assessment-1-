import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prepare_bank_data(filepath: str, student_id: int):
    """
    Loads, prepares, and splits the Bank Marketing dataset according to ECOM6004 requirements.
    """
    df = pd.read_csv(filepath)
    
    # Required transformations
    df['target'] = (df['y'] == 'yes').astype(int)
    df['age10'] = (df['age'] - 40) / 10.0
    df['balance1000'] = df['balance'] / 1000.0
    
    # Construct prior_status
    def get_prior_status(row):
        if row['pdays'] == -1:
            return 'not previously contacted'
        elif row['poutcome'] == 'success':
            return 'previous success'
        elif row['poutcome'] == 'failure':
            return 'previous failure'
        else:
            return 'previous other/unknown'
            
    df['prior_status'] = df.apply(get_prior_status, axis=1)
    
    # Categorical Type Casting and Reference Category Setting
    df['housing'] = pd.Categorical(df['housing'], categories=['no', 'yes'], ordered=False)
    df['loan'] = pd.Categorical(df['loan'], categories=['no', 'yes'], ordered=False)
    df['contact'] = pd.Categorical(df['contact'], categories=['telephone', 'cellular', 'unknown'], ordered=False)
    df['prior_status'] = pd.Categorical(
        df['prior_status'], 
        categories=['not previously contacted', 'previous success', 'previous failure', 'previous other/unknown'],
        ordered=False
    )
    
    # Stratified 80/20 Train/Test Split using Student ID as Seed
    train, test = train_test_split(
        df, test_size=0.20, random_state=student_id, stratify=df['target']
    )
    
    return df, train.copy(), test.copy()
