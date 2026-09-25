import pandas as pd
from sklearn.model_selection import train_test_split

def audit_data(df: pd.DataFrame):
    """
    Audits the Bank Marketing dataset for data quality issues.
    Returns audit summary as pandas DataFrame.
    """
    # Basic statistics
    audit_results = {
        'total_observations': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'target_distribution': df['y'].value_counts().to_dict(),
        'response_level_distribution': df['response_level'].value_counts().sort_index().to_dict(),
        'age_range': f"{df['age'].min()} - {df['age'].max()}",
        'balance_range': f"{df['balance'].min()} - {df['balance'].max()}",
    }
    
    # Value range checks for numeric variables
    audit_results['age_valid_range'] = 'PASS' if (df['age'].min() >= 18 and df['age'].max() <= 100) else 'FAIL'
    audit_results['balance_valid_range'] = 'PASS' if df['balance'].min() >= -10000 else 'FAIL'
    audit_results['duration_valid_range'] = 'PASS' if (df['duration'].min() >= 0 and df['duration'].max() <= 5000) else 'FAIL'
    audit_results['campaign_valid_range'] = 'PASS' if (df['campaign'].min() >= 1 and df['campaign'].max() <= 100) else 'FAIL'
    audit_results['pdays_valid_range'] = 'PASS' if (df['pdays'].min() == -1 and df['pdays'].max() <= 1000) else 'FAIL'
    
    # Category validation for categorical variables
    valid_jobs = {'admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 
                  'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'}
    audit_results['job_valid_categories'] = 'PASS' if set(df['job'].unique()).issubset(valid_jobs) else 'FAIL'
    
    valid_marital = {'married', 'divorced', 'single'}
    audit_results['marital_valid_categories'] = 'PASS' if set(df['marital'].unique()).issubset(valid_marital) else 'FAIL'
    
    valid_education = {'unknown', 'secondary', 'primary', 'tertiary'}
    audit_results['education_valid_categories'] = 'PASS' if set(df['education'].unique()).issubset(valid_education) else 'FAIL'
    
    valid_default = {'yes', 'no'}
    audit_results['default_valid_categories'] = 'PASS' if set(df['default'].unique()).issubset(valid_default) else 'FAIL'
    
    valid_housing = {'yes', 'no'}
    audit_results['housing_valid_categories'] = 'PASS' if set(df['housing'].unique()).issubset(valid_housing) else 'FAIL'
    
    valid_loan = {'yes', 'no'}
    audit_results['loan_valid_categories'] = 'PASS' if set(df['loan'].unique()).issubset(valid_loan) else 'FAIL'
    
    valid_contact = {'unknown', 'telephone', 'cellular'}
    audit_results['contact_valid_categories'] = 'PASS' if set(df['contact'].unique()).issubset(valid_contact) else 'FAIL'
    
    valid_poutcome = {'unknown', 'other', 'failure', 'success'}
    audit_results['poutcome_valid_categories'] = 'PASS' if set(df['poutcome'].unique()).issubset(valid_poutcome) else 'FAIL'
    
    # Consistency check: response_level > 0 iff y = "yes"
    consistency_check = (df['response_level'] > 0) == (df['y'] == 'yes')
    audit_results['y_response_level_consistency'] = 'PASS' if consistency_check.all() else 'FAIL'
    audit_results['y_response_level_inconsistencies'] = (~consistency_check).sum()
    
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
