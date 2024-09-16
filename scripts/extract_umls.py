from scispacy.candidate_generation import CandidateGenerator
import pandas as pd
from pathlib import Path
from tqdm import tqdm

generator = CandidateGenerator(name='umls')


def get_candidates(
        generator, target, k=30, threshold=0.5, no_definition_threshold=0.95, 
        filter_for_definitions=True, max_entities_per_mention=5):
    """ 
    Given a text and a target, return the UMLS entities that match the target
    """
    # We can use the CandidateGenerator to get the UMLS entities

    candidates = generator([target], k)[0]
    predicted = []
    for cand in candidates:
        score = max(cand.similarities)
        if (
            filter_for_definitions
            and generator.kb.cui_to_entity[cand.concept_id].definition is None
            and score < no_definition_threshold
        ):
            continue
        if score > threshold:
            name = cand.canonical_name if hasattr(cand, 'canonical_name') else cand.aliases[0]
            predicted.append((cand.concept_id, name, score))
    sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[2])
    return target, sorted_predicted[: max_entities_per_mention]


def run_extraction(predictions, pmcids=None):
    results = []
    for _, doc_preds in tqdm(predictions.groupby('pmcid')):
        for ix, pred in doc_preds.iterrows():
            # Get the UMLS entities that match the targettarg
            start_char = pred['start_char'] if 'start_char' in pred else None
            end_char = pred['end_char'] if 'end_char' in pred else None
            if pred['group_name'] == 'patients' and pd.isna(pred['diagnosis']) == False:
                resolved_target, target_ents = get_candidates(
                    generator, pred['diagnosis'], start_char=start_char, end_char=end_char)

                for ent in target_ents:
                    results.append({
                        "pmcid": int(doc['pmcid']),
                        "diagnosis": resolved_target,
                        "umls_cui": ent[0],
                        "umls_name": ent[1],
                        "umls_prob": ent[2],
                        "count": pred['count'],
                        "group_ix": ix,
                        "start_char": start_char,
                        "end_char": end_char,
                    })

    return results


# Apply to all predictions, with different sources
input_predictions = [
    ('md', 'full_md_demographics-zeroshot_gpt-4o-mini-2024-07-18_noabbrev.csv'),
    ('md', 'full_md_demographics-zeroshot_gpt-4o-2024-05-13_noabbrev.csv'),
    ('html', 'full_html_demographics-zeroshot_gpt-4o-mini-2024-07-18_noabbrev.csv'),
    ('md', 'chunked_demographics-zeroshot_gpt-4o-2024-05-13_minc-40_maxc-4000_noabbrev.csv')
    ('md', 'chunked_demographics-zeroshot_gpt-4o-mini-2024-07-18_minc-40_maxc-4000_noabbrev.csv')
    ('md', 'full_md_demographics-zeroshot-ftstrict_gpt-4o-mini-2024-07-18_noabbrev.csv'),
    ('md', 'full_md_demographics-zeroshot-ftstrict_gpt-4o-2024-05-13_noabbrev.csv')
]

results_dir = Path('../outputs/extractions')

for source, pred_path in input_predictions:
    predictions = pd.read_csv(results_dir / pred_path)
    results = run_extraction(predictions)
    results_df = pd.DataFrame(results)

    # Remove _clean from the filename
    out_name = pred_path.replace('_noabbrev', '_umls')
    results_df.to_csv(results_dir / out_name, index=False)
