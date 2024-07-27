from typing import List

from mage_ai.data_cleaner.transformer_actions.base import BaseAction
from mage_ai.data_cleaner.transformer_actions.constants import ActionType, Axis
from mage_ai.data_cleaner.transformer_actions.utils import build_transformer_action
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlops_zoomcamp_project.utils.preprocessing_file import apply_one_hot_encoding


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def apply_preprocessing_into_dataset(data: pd.DataFrame) -> pd.DataFrame:

    data = data.drop(columns=['id', 'CustomerId', 'Surname'])

    # Convert data types
    data['Age'] = data['Age'].astype(int)
    data['HasCrCard'] = data['HasCrCard'].astype(int)
    data['IsActiveMember'] = data['IsActiveMember'].astype(int)

    # One-hot encode categorical variables
    data = pd.get_dummies(
        data, 
        columns=['Geography', 'Gender'], 
        drop_first=True
    )

    scaler = StandardScaler()
    numerical_features = ['CreditScore', 'Balance', 'EstimatedSalary']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data


@transformer
def execute_transformer_action(
    df: pd.DataFrame, 
    *args, 
    **kwargs
) -> pd.DataFrame:
    """
    Execute Transformer Action: ActionType.REMOVE

    Docs: https://docs.mage.ai/guides/transformer-blocks#remove-columns
    """
    df = apply_preprocessing_into_dataset(df)

    return df
    

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
