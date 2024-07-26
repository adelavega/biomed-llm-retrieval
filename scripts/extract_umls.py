import spacy
from scispacy.candidate_generation import CandidateGenerator
from scispacy.abbreviation import AbbreviationDetector
import pandas as pd
from pathlib import Path
from labelrepo import database
from tqdm import tqdm

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("abbreviation_detector")

generator = CandidateGenerator(name='umls')


def load_docs(pmids, source='md'):
    if source == 'md':
        docs = pd.read_sql(
            "select pmcid, text from document",
            database.get_database_connection(),
        )
        docs = docs[
            docs.pmcid.isin(pmids)].to_dict(orient='records')

    elif source == 'html':
        docs = pd.read_csv('../data/html_combined.csv')
        docs = docs[docs.pmcid.isin(pmids)].to_dict(orient='records')

    return docs


def get_candidates(
        generator, processed_doc, target, resolve_abbreviations=True, start_char=None, end_char=None,
        k=30, threshold=0.5, no_definition_threshold=0.95, filter_for_definitions=True, max_entities_per_mention=5):
    """ Given a text and a target, return the UMLS entities that match the target
    Takes advantage of abbreciation detection from full text and entity linking to UMLS.
    """
    # First we need to resolve abbreciation in the target text
    if resolve_abbreviations:
        for abrv in processed_doc._.abbreviations:
            if abrv.text in target:
                # If start and end char are provided, only resolve abbreviations within the target
                if start_char is not None and end_char is not None:
                    if not (abrv.start_char >= start_char and abrv.end_char <= end_char):
                        continue 
                target = target.replace(abrv.text, abrv._.long_form.text)

    # Second we can use the CandidateGenerator to get the UMLS entities

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


def run_extraction(docs, predictions, pmcids=None):
    if pmcids is not None:
        docs = [d for d in docs if int(d['pmcid']) in pmcids]

    results = []
    for doc in tqdm(docs):
        doc_preds = predictions[predictions.pmcid == doc['pmcid']]
        processed_doc = nlp(doc['text'])
        for ix, pred in doc_preds.iterrows():
            # Get the UMLS entities that match the targettarg
            start_char = pred['start_char'] if 'start_char' in pred else None
            end_char = pred['end_char'] if 'end_char' in pred else None
            if pred['group_name'] == 'patients' and pd.isna(pred['diagnosis']) == False:
                resolved_target, target_ents = get_candidates(
                    generator, processed_doc, pred['diagnosis'], start_char=start_char, end_char=end_char)

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
    # ('md', 'full_md_demographics-zeroshot_gpt-4o-mini-2024-07-18_clean.csv'),
    # ('md', 'full_md_demographics-zeroshot_gpt-4o-2024-05-13_clean.csv'),
    # ('html', 'full_html_demographics-zeroshot_gpt-4o-mini-2024-07-18_clean.csv'),
    # ('md', 'chunked_demographics-zeroshot_gpt-4o-2024-05-13_minc-40_maxc-4000_clean.csv')
    ('md', 'chunked_demographics-zeroshot_gpt-4o-mini-2024-07-18_minc-40_maxc-4000_clean.csv')

]

output_dir = Path('../outputs')

for source, pred_path in input_predictions:
    predictions = pd.read_csv(output_dir / pred_path)
    docs = load_docs(predictions.pmcid.unique(), source)
    results = run_extraction(docs, predictions)
    results_df = pd.DataFrame(results)

    # Remove _clean from the filename
    out_name = pred_path.replace('_clean', '_umls')
    results_df.to_csv(output_dir / out_name, index=False)