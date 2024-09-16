""" Uses scispacy to replace abbreviations from LLM outputs
For now only applies to "diagnosis" field
 """
import spacy
from scispacy.abbreviation import AbbreviationDetector
from labelrepo import database
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("abbreviation_detector")

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

    # Turn into dictionary with key as pmcid
    docs = {doc['pmcid']: doc for doc in docs}
    return docs


def replace_abbreviations(doc, target, start_char=None, end_char=None, remove_parenth=True):
    """
    Replace abbreviations in the target string with their long form

    Args:
        doc: spacy doc object
        target: target string to replace abbreviations in
        start_char: start char of the target in the doc
        end_char: end char of the target in the doc
        remove_parenth: whether to remove content in parentheses after the abbreviation
    """
    for abrv in doc._.abbreviations:
        if abrv.text in target and not abrv.text in abrv._.long_form.text:
            # If start and end char are provided, only resolve abbreviations within the target
            if start_char is not None and end_char is not None:
                if not (abrv.start_char >= start_char and abrv.end_char <= end_char):
                    continue 

            if remove_parenth:
                # If abbreviation is enclosed in parentheses, remove the content in parentheses
                target = re.sub(rf'\({abrv.text}\)', '', target)
                
            target = target.replace(abrv.text, abrv._.long_form.text)

    return target.strip()


def run_abbrev(docs, predictions):
    for pmcid, preds in tqdm(predictions.groupby('pmcid'), total=len(predictions.pmcid.unique())):
        processed_doc = nlp(docs[pmcid]['text'])

        for ix, pred in preds.iterrows():
            # Get the UMLS entities that match the targettarg
            start_char = pred['start_char'] if 'start_char' in pred else None
            end_char = pred['end_char'] if 'end_char' in pred else None

            
            if pred['group_name'] == 'patients' and pd.isna(pred['diagnosis']) == False:
                target_noabbrev = replace_abbreviations(
                    processed_doc, pred['diagnosis'], start_char=start_char, end_char=end_char)
                
                if target_noabbrev != pred['diagnosis']:
                    predictions.loc[ix, 'diagnosis'] = target_noabbrev
                    continue

    return predictions


# Apply to all predictions, with different sources
input_predictions = [
    ('md', 'full_md_demographics-zeroshot_gpt-4o-mini-2024-07-18_clean.csv'),
    ('md', 'full_md_demographics-zeroshot_gpt-4o-2024-05-13_clean.csv'),
    ('html', 'full_html_demographics-zeroshot_gpt-4o-mini-2024-07-18_clean.csv'),
    ('md', 'chunked_demographics-zeroshot_gpt-4o-2024-05-13_minc-40_maxc-4000_clean.csv'),
    ('md', 'chunked_demographics-zeroshot_gpt-4o-mini-2024-07-18_minc-40_maxc-4000_clean.csv'),
    ('md', 'full_md_demographics-zeroshot-ftstrict_gpt-4o-mini-2024-07-18_clean.csv'),
    ('md', 'full_md_demographics-zeroshot-ftstrict_gpt-4o-2024-05-13_clean.csv')
]

results_dir = Path('../outputs/')
extractions_dir = results_dir / 'extractions'

for source, pred_path in input_predictions:
    print(f'Processing {pred_path}')
    predictions = pd.read_csv(extractions_dir / pred_path)
    docs = load_docs(predictions.pmcid.unique(), source)
    predictions = run_abbrev(docs, predictions)

    # Remove _clean from the filename
    out_name = pred_path.replace('_clean', '_noabbrev')
    predictions.to_csv(results_dir / out_name, index=False)
