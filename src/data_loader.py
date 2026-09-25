import pandas as pd
from sklearn.model_selection import train_test_split

def audit_data(df: pd.DataFrame):
    """
    Audits the Bank Marketing dataset for data quality issues.
    Returns audit summary as pandas DataFrame.
    """
    audit_results = {
        'total_observations': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'target_distribution': df['y'].value_counts().to_dict(),
        'response_level_distribution': df['response_level'].value_counts().sort_index().to_dict(),
        'age_range': f"{df['age'].min()} - {df['age'].max()}",
        'balance_range': f"{df['balance'].min()} - {df['balance'].max()}",
    }
    return pd.DataFrame([audit_results]).T

def load_and_prepare_bank_data(filepath: str, student_id: int):
    """
    Loads, prepares, and splits the Bank Marketing dataset according to ECOM6004 requirements.
    
    Empirical Setting: Bank Marketing - "Which clients should be contacted?"
    - Binary outcome: y (yes/no for term deposit subscription)
    - Ordered outcome: response_level (0=no subscription, 1=subscription after 2+ contacts, 2=subscription on first contact)
    """
    df = pd.read_csv(filepath)
    
    # Audit data before preparation
    audit_summary = audit_data(df)
    
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
    
    return df, train.copy(), test.copy(), audit_summary
