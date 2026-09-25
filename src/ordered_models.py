import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

def fit_ordered_models(df: pd.DataFrame):
    """
    Fits Ordered Logit and Ordered Probit models using the full dataset.
    Note: OrderedModel design matrix should NOT include an intercept.
    """
    # Create design matrix without intercept
    X = pd.get_dummies(
        df[['age10', 'balance1000', 'housing', 'loan', 'contact', 'previous', 'prior_status']], 
        drop_first=True
    ).astype(float)
    
    y = df['response_level'].astype(int)
    
    ologit = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)
    oprobit = OrderedModel(y, X, distr='probit').fit(method='bfgs', disp=False)
    
    return {'ologit': ologit, 'oprobit': oprobit}
