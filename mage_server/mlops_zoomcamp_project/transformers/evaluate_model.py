from typing import List
import joblib

import numpy as np
import pandas as pd
import random
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(list_df: List[pd.DataFrame], clf, *args, **kwargs):
    
    X_train, y_train, X_val, y_val = list_df
    y_pred = clf.predict(X_val)

    y_pred = clf.predict(X_val)
    roc_auc = roc_auc_score(y_val, y_pred)

    try:
        classification_report_result = classification_report(
            y_val,
            y_pred,
            output_dict=True,
            # target_names=target_names,
        )
    except ValueError as err:
        print(f'Error occurred during classification report: {err}')
        classification_report_result = ''

    metrics = dict(
        classification_report=classification_report_result,
        roc_auc=roc_auc,
    )

    return metrics