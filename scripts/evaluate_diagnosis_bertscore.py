from pathlib import Path
import pandas as pd
from bert_score import BERTScorer
from itertools import product
from tqdm import tqdm

# Load annotations
combined_annotations = pd.read_csv('../annotations/combined_pd.csv')

subset_cols = ['diagnosis', 'group_name', 'subgroup_name', 'count', 'pmcid']
combined_annotations = combined_annotations[subset_cols].sort_values('pmcid')

# Replace column names space with underscore
combined_annotations.columns = combined_annotations.columns.str.replace(' ', '_')

output_dir = Path('../outputs/demographics')
results_dir = output_dir / 'extractions'

scorer = BERTScorer(lang='en-sci', rescale_with_baseline=True)

## Matching score approach
## First only look at imaging samples and groups with diaganosis
## Which also should be "group_name" == "patients"
## If there are still multiple groups, we will try to match the two most likely groups
## By taking the maximum similarity for first group and then removing that group
## And then taking the maximum similarity for the second group

def _filter_imaging_sample(x):
    # If multiple vaules for imaging_sample, take those != no
    if x.imaging_sample.unique().size > 1:
        return x[x.imaging_sample != 'no']
    return x


def _evaluate(annotations, predictions, agg=True):
    " Evaluate the predictions against the annotations  for the diagnosis column"
        # Subset to only include mri participant groups
    if 'imaging_sample' in predictions.columns:
        predictions = predictions.groupby('pmcid')[predictions.columns].apply(
            lambda x: _filter_imaging_sample(x)
        ).reset_index(drop=True)

    if 'assessment_type' in predictions.columns:
        predictions = predictions[predictions.assessment_type != 'behavioral']


    # Subset to only include patients
    annotations = annotations[annotations.group_name == 'patients']
    predictions = predictions[predictions.pmcid.isin(annotations.pmcid.unique())]
    predictions = predictions[predictions.group_name == 'patients']

    all_scores = []

    for pmcid, preds in predictions.groupby('pmcid'):
        annots = annotations[annotations.pmcid == pmcid]

        # Get bert scores for all combinations of the preds/annotations
        scores = []
        for cand_ix, ref_ix in list(product(range(preds.shape[0]), range(annots.shape[0]))):
            if pd.isna(preds.iloc[cand_ix].diagnosis) or pd.isna(annots.iloc[ref_ix].diagnosis):
                scores.append((cand_ix, ref_ix, 0, 0, -1))
                continue
            # Get the similarity score
            score = scorer.score(
                [preds.iloc[cand_ix].diagnosis],
                [annots.iloc[ref_ix].diagnosis]
            )
            scores.append((cand_ix, ref_ix) + score)

        # Starting with the highest f1 score, take the matched pairs
        # If that annotation or prediction group has not been matched
        cand_matched = []
        ref_matched = []
        for score in sorted(scores, key=lambda x: x[-1], reverse=True):
            cand_ix, ref_ix, p, r, f1 = score
            if f1 == -1:
                p, r, f1 = (pd.NA, pd.NA, pd.NA)  # If f1 is -1, set to NA to indicate missing value
            if cand_ix not in cand_matched and ref_ix not in ref_matched:
                cand_matched.append(cand_ix)
                ref_matched.append(ref_ix)
                all_scores.append({
                    'pmcid': pmcid,
                    'prediction': preds.iloc[cand_ix].diagnosis,
                    'annotation': annots.iloc[ref_ix].diagnosis,
                    'p': p,
                    'r': r,
                    'f1': f1
                })

    if agg:
        all_scores = pd.DataFrame(all_scores)
        all_scores = all_scores.agg({
            'p': 'mean',
            'r': 'mean',
            'f1': 'mean'
        }).reset_index()

    return all_scores


all_files = list(results_dir.glob('chunked_*zeroshot*_clean.csv')) + list(results_dir.glob('full_*zeroshot*_clean.csv'))
eval_results = []
for f in tqdm(all_files):
    predictions = pd.read_csv(f)
    predictions.columns = predictions.columns.str.replace(' ', '_')

    stats = _evaluate(combined_annotations, predictions)

    stats = pd.DataFrame(stats).reset_index()

    # Add metadata to pd dataframe
    fsplit = f.stem.split('_')
    if fsplit[0] == 'chunked':
        strategy, task, model_name, min_chars, max_chars, _ = fsplit
        stats['min_chars'] = min_chars.split('-')[1]
        stats['max_chars'] = max_chars.split('-')[1]
    else: 
        strategy, source, task, model_name, _ = fsplit
        stats['source'] = source

    stats['strategy'] = strategy
    stats['task'] = task
    stats['model_name'] = model_name

    eval_results.append(stats)

eval_results = pd.concat(eval_results)
eval_results = pd.DataFrame(eval_results)
eval_results.rename(columns={'index': 'metric', '0': 'score'}, inplace=True)
eval_results.to_csv(output_dir / 'bertscore_results.csv', index=False)