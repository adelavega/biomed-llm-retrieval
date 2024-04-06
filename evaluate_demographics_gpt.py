import numpy as np
from pathlib import Path
import pandas as pd
import glob
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

results_dir = Path('outputs')


def _evaluate(annotations, predictions):
    # Overall recall
    # Print both fraction and percentage
    n_studies_in_predictions = len(set(predictions.pmcid.unique()))
    n_studies_in_annotations = len(set(annotations.pmcid.unique()))
    print(f"Studies with prediction: {n_studies_in_predictions / n_studies_in_annotations:.2f} ({n_studies_in_predictions} / {n_studies_in_annotations})")

    # Subset to only pmcids in predictions
    diff = set(set(annotations.pmcid.unique()) - set(predictions.pmcid.unique()))
    annotations = annotations[annotations.pmcid.isin(predictions.pmcid.unique())]

    # Match compare
    match_compare = hungarian_match_compare(annotations, predictions)

    # Compare by columns (matched accuracy)
    res_mean, res_sums, counts = score_columns(annotations, predictions)

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

    combined_stats = {
        "matched_accuracy": match_compare,
        "counts": counts,
        "avg_mean_percentage_error": res_mean,
        "summed_mean_percentage_error": res_sums,
    }

    print(pd.DataFrame(combined_stats))
    return combined_stats



for f in sorted(results_dir.glob('eval_participant_demographics_*_clean.csv')):
    predictions = pd.read_csv(f)
    print(f"\nResults for {f.name}")
    _evaluate(jerome_pd, predictions)


# for f in glob.glob('../fmri_participant_demographics/data/outputs/gpt/eval_*_clean.csv'):
#     predictions = pd.read_csv(f)
#     print(f"\nResults for {f}")
#     _evaluate(jerome_pd, predictions)