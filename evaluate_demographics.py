from pathlib import Path
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

# Replace column names space with underscore
jerome_pd.columns = jerome_pd.columns.str.replace(' ', '_')

results_dir = Path('outputs')


def _filter_imaging_sample(x):
    # If multiple vaules for imaging_sample, take those != no
    if x.imaging_sample.unique().size > 1:
        return x[x.imaging_sample != 'no']
    return x


def _evaluate(annotations, predictions):
    # Overall recall
    # Print both fraction and percentage
    n_studies = len(set(predictions.pmcid.unique()))

    # Subset to only include mri participant groups
    if 'imaging_sample' in predictions.columns:
        predictions = predictions.groupby('pmcid').apply(
            lambda x: _filter_imaging_sample(x)
        ).reset_index(drop=True)

    # Subset to only pmcids in predictions
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

    combined_stats = {
        "hungarian_matched_error": match_compare,
        "counts": counts,
        "avg_mean_percentage_error": res_mean,
        "summed_mean_percentage_error": res_sums,
    }

    return (
        n_studies, correct_n_groups.mean(), more_groups_pred.mean(),
        less_groups_pred.mean(), combined_stats
    )


all_results = []
for f in sorted(results_dir.glob('eval_*_clean.csv')):
    predictions = pd.read_csv(f)
    predictions.columns = predictions.columns.str.replace(' ', '_')

    n_studies, corr_n_groups, more, less, stats = _evaluate(
        jerome_pd, predictions)

    # Add metadata to pd dataframe
    _, task, model_name, min_chars, max_chars, _ = f.stem.split('_')

    stats = pd.DataFrame(stats).reset_index()
    stats = stats.rename(columns={'index': 'variable'})
    stats['n_studies'] = n_studies
    stats['task'] = task
    stats['model_name'] = model_name
    stats['min_chars'] = min_chars.split('-')[1]
    stats['max_chars'] = max_chars.split('-')[1]
    stats['corr_groups'] = corr_n_groups
    stats['more_groups'] = more
    stats['less_groups'] = less

    all_results.append(stats)

all_results = pd.concat(all_results)
all_results = pd.DataFrame(all_results)
all_results.to_csv(results_dir / 'all_results.csv', index=False)
