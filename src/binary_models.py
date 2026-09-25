import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, brier_score_loss

FORMULA = "target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)"

def fit_binary_models(train_df: pd.DataFrame):
    """Fits Linear Probability Model, Logit, and Probit models on the training sample."""
    lpm = smf.ols(FORMULA, data=train_df).fit()
    logit = smf.logit(FORMULA, data=train_df).fit(disp=False)
    probit = smf.probit(FORMULA, data=train_df).fit(disp=False)
    
    return {'lpm': lpm, 'logit': logit, 'probit': probit}

def evaluate_test_set(model, test_df: pd.DataFrame, threshold: float = 0.50):
    """Evaluates selected binary model once on the held-out test set."""
    preds = model.predict(test_df)
    pred_labels = (preds >= threshold).astype(int)
    actuals = test_df['target']
    
    cm = confusion_matrix(actuals, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    bal_acc = (sens + spec) / 2.0
    brier = brier_score_loss(actuals, preds)
    
    metrics = {
        'obs_event_rate': actuals.mean(),
        'mean_pred_prob': preds.mean(),
        'brier_score': brier,
        'sensitivity': sens,
        'specificity': spec,
        'balanced_accuracy': bal_acc,
        'confusion_matrix': cm
    }
    return metrics
