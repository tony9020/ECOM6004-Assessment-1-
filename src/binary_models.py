import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, brier_score_loss

FORMULA = "target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)"

def fit_binary_models(train_df: pd.DataFrame):
    """
    Fits Linear Probability Model, Logit, and Probit models on the training sample.
    
    Empirical Setting: Predicting term deposit subscription (y) to determine which clients should be contacted.
    """
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

def create_decision_log(test_metrics: dict, model_name: str = 'logit'):
    """
    Creates a binary-stage decision log translating test evidence into a cautious client decision.
    Returns decision log as pandas DataFrame.
    """
    decision_log = {
        'model_used': model_name,
        'test_brier_score': test_metrics['brier_score'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'test_sensitivity': test_metrics['sensitivity'],
        'test_specificity': test_metrics['specificity'],
        'observed_event_rate': test_metrics['obs_event_rate'],
        'mean_predicted_probability': test_metrics['mean_pred_prob'],
        'decision': 'Proceed with targeted contact strategy' if test_metrics['balanced_accuracy'] > 0.6 else 'Caution: model performance moderate, use with additional business judgment',
        'rationale': f"Model achieves {test_metrics['balanced_accuracy']:.2%} balanced accuracy with Brier score of {test_metrics['brier_score']:.4f}. "
                    f"Sensitivity ({test_metrics['sensitivity']:.2%}) and specificity ({test_metrics['specificity']:.2%}) indicate "
                    f"{'acceptable' if test_metrics['balanced_accuracy'] > 0.6 else 'moderate'} discrimination for client targeting."
    }
    return pd.DataFrame([decision_log]).T
