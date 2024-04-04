import pprint
import pandas as pd
from labelrepo.projects.participant_demographics import get_participant_demographics
from publang.evaluate import score_columns, hungarian_match_compare


# Load annotations
subgroups = get_participant_demographics(include_locations=True)
jerome_pd = subgroups[(subgroups.project_name == 'participant_demographics') & \
                      (subgroups.annotator_name == 'Jerome_Dockes')]
subset_cols = ['count', 'diagnosis', 'group_name', 'subgroup_name', 'male count',
       'female count', 'age mean', 'age minimum', 'age maximum',
       'age median', 'pmcid']
jerome_pd = jerome_pd[subset_cols].sort_values('pmcid')

predictions_default = pd.read_csv('data/jerome_subset_1_predictions.csv')
predictions_smaller_chunks = pd.read_csv('data/jerome_subset_1_smaller_chunks_predictions.csv')
predictions_1000_chunks = pd.read_csv('data/participant_demographics_gpt_maxtokens-1000.csv')


def _evaluate(annotations, predictions):

    # Match compare
    match_compare = hungarian_match_compare(annotations, predictions)

    # Compare by columns (matched accuracy)
    res_mean, res_sums, counts = score_columns(annotations, predictions)

    # Subset to only pmcids in predictions
    diff = set(set(annotations.pmcid.unique()) - set(predictions.pmcid.unique()))
    annotations = annotations[annotations.pmcid.isin(predictions.pmcid.unique())]

    # Compute overlap of pmcids
    pred_n_groups = predictions.groupby('pmcid').size()
    n_groups = annotations.groupby('pmcid').size()
    correct_n_groups = (n_groups == pred_n_groups)
    more_groups_pred = (n_groups < pred_n_groups)
    less_groups_pred = (n_groups > pred_n_groups)

    print(f"Exact match # of groups: {correct_n_groups.mean():.2f}\n",
    f"More groups predicted: {more_groups_pred.mean():.2f}\n",
    f"Less groups predicted: {less_groups_pred.mean():.2f}\n",
    f"Missing pmcids: {diff}\n"
    )

    # Compare by columns (matched accuracy)
    print("Column wise comparison of predictions and annotations (error):\n")
    pprint.pprint(match_compare)

    # Compare by columns (count of pmcids with overlap)
    print("\nPercentage response given by pmcid:\n")
    pprint.pprint(counts)


    # Compare by columns (summed mean percentage error on sum by pmcid)
    print("\nSummed Mean percentage error:\n")
    pprint.pprint(res_mean)

    # Compare by columns (summed mean percentage error on mean by pmcid)
    print("\nAveraged Mean percentage error:\n")
    pprint.pprint(res_sums)


print("Evaluate with default chunk size (4000)")
_evaluate(jerome_pd, predictions_default)

print("Evaluate with smaller chunk size (2000)")
_evaluate(jerome_pd, predictions_smaller_chunks)

print("Evaluate with smaller chunk size (1000)")
_evaluate(jerome_pd, predictions_1000_chunks)
