import datetime

import pandas as pd
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


num_features = [
	'CreditScore',
	'Age',
	'Tenure',
	'Balance',
	'NumOfProducts',
	'EstimatedSalary'
]
cat_features = [
	'Geography',
	'Gender',
	'HasCrCard',
	'IsActiveMember',
]

column_mapping = ColumnMapping(
	prediction='Exited',
	numerical_features=num_features,
	categorical_features=cat_features,
	target=None
)

report = Report(metrics = [
	ColumnDriftMetric(column_name='Exited'),
	DatasetDriftMetric(),
	DatasetMissingValuesMetric()
])


@transformer
def transform(reference_data, current_data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    report.run(
		reference_data=reference_data,
		current_data=current_data,
		column_mapping=column_mapping,
	)
    report_result = report.as_dict()

    timestamp = datetime.datetime.now()
    prediction_drift = str(
		report_result['metrics'][0]['result']['drift_score']
	)
    num_drifted_columns = str(
		report_result['metrics'][1]['result']['number_of_drifted_columns']
	)
    share_missing_values = str(
		report_result['metrics'][2]['result']['current']['share_of_missing_values']
	)

    dummy_metrics_dict = {
		"timestamp": timestamp,
		"prediction_drift": prediction_drift,
		"num_drifted_columns": num_drifted_columns,
		"share_missing_values": share_missing_values,
	}

    return_data = pd.DataFrame(dummy_metrics_dict, index=[0])

    return return_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'

