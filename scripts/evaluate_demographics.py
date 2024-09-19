from pathlib import Path
import pandas as pd
from publang.evaluate import score_columns, hungarian_match_compare

# Load annotations
combined_annotations = pd.read_csv('../annotations/combined_pd.csv')

subset_cols = ['count', 'diagnosis', 'group_name', 'subgroup_name', 'male count',
       'female count', 'age mean', 'age minimum', 'age maximum',
       'age median', 'pmcid']
combined_annotations = combined_annotations[subset_cols].sort_values('pmcid')

# Replace column names space with underscore
combined_annotations.columns = combined_annotations.columns.str.replace(' ', '_')

outout_dir = Path('../outputs')
results_dir = output_dir / 'extractions'


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
        predictions = predictions.groupby('pmcid')[predictions.columns].apply(
            lambda x: _filter_imaging_sample(x),
        ).reset_index(drop=True)

    if 'assessment_type' in predictions.columns:
        predictions = predictions[predictions.assessment_type != 'behavioral']

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


eval_results = []
for f in sorted(results_dir.glob('chunked_*_clean.csv')):
    predictions = pd.read_csv(f)
    predictions.columns = predictions.columns.str.replace(' ', '_')

    n_studies, corr_n_groups, more, less, stats = _evaluate(
        combined_annotations, predictions)

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

    eval_results.append(stats)

eval_results = pd.concat(eval_results)
eval_results = pd.DataFrame(eval_results)
eval_results.to_csv(results_dir / 'chunked_results.csv', index=False)

# For extraction from full-text (html or md)
full_results = []
for f in sorted(results_dir.glob('full_*_clean.csv')):
    predictions = pd.read_csv(f)
    predictions.columns = predictions.columns.str.replace(' ', '_')

    n_studies, corr_n_groups, more, less, stats = _evaluate(
        combined_annotations, predictions)

    # Add metadata to pd dataframe
    _, source, task, model_name, _ = f.stem.split('_')

    if source == 'html':
        html_pmids = predictions.pmcid.unique()

    stats = pd.DataFrame(stats).reset_index()
    stats = stats.rename(columns={'index': 'variable'})
    stats['n_studies'] = n_studies
    stats['task'] = task
    stats['model_name'] = model_name
    stats['corr_groups'] = corr_n_groups
    stats['more_groups'] = more
    stats['less_groups'] = less
    stats['source'] = source
    stats['subset'] = 'full'

    full_results.append(stats)

# Re-compute for MD on html subset
for f in sorted(results_dir.glob('full_*md*_clean.csv')):
    predictions = pd.read_csv(f)
    predictions.columns = predictions.columns.str.replace(' ', '_')

    # Subset using HTML_PMIDS
    predictions = predictions[predictions.pmcid.isin(html_pmids)] 

    n_studies, corr_n_groups, more, less, stats = _evaluate(
        combined_annotations, predictions)

    # Add metadata to pd dataframe
    _, source, task, model_name, _ = f.stem.split('_')

    stats = pd.DataFrame(stats).reset_index()
    stats = stats.rename(columns={'index': 'variable'})
    stats['n_studies'] = n_studies
    stats['task'] = task
    stats['model_name'] = model_name
    stats['corr_groups'] = corr_n_groups
    stats['more_groups'] = more
    stats['less_groups'] = less
    stats['source'] = source
    stats['subset'] = 'html_match'

    full_results.append(stats)


full_results = pd.concat(full_results)
full_results = pd.DataFrame(full_results)
full_results.to_csv(results_dir / 'full_results.csv', index=False)
