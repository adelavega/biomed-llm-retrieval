import spacy
from scispacy.candidate_generation import CandidateGenerator
from scispacy.abbreviation import AbbreviationDetector
import pandas as pd
from pathlib import Path
from labelrepo import database
from tqdm import tqdm

# Load predictions
base_name = 'eval_demographics-fewshot_gpt-4o-2024-05-13_minc-40_maxc-4000'
gpt_predictions = pd.read_csv(f'../outputs/{base_name}_clean.csv')

# Load articles that have been annotated
docs = pd.read_sql(
    "select pmcid, text from document",
    database.get_database_connection(),
)
docs = docs[
    docs.pmcid.isin(gpt_predictions.pmcid)].to_dict(orient='records')

output_dir = Path('../outputs')

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("abbreviation_detector")

generator = CandidateGenerator(name='umls')


def get_candidates(
		generator, processed_doc, target, resolve_abbreviations=True, start_char=None, end_char=None,
		k=30, threshold=0.5, no_definition_threshold=0.95, filter_for_definitions=True, max_entities_per_mention=5):
	""" Given a text and a target, return the UMLS entities that match the target
	Takes advantage of abbreciation detection from full text and entity linking to UMLS.
	"""
	# First we need to resolve abbreciation in the target text
	if resolve_abbreviations:
		for abrv in processed_doc._.abbreviations:
			if abrv.start_char >= start_char and abrv.end_char <= end_char:
				if abrv.text in target:
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
	

def run_extraction(docs, pmcids=None):
	if pmcids is not None:
		docs = [d for d in docs if int(d['pmcid']) in pmcids]
		
	results = []
	for doc in tqdm(docs):
		doc_preds = gpt_predictions[gpt_predictions.pmcid == doc['pmcid']]
		processed_doc = nlp(doc['text'])
		for ix, pred in doc_preds.iterrows():
			# Get the UMLS entities that match the targettarg
			if pred['group_name'] == 'patients':
				resolved_target, target_ents = get_candidates(
					generator, processed_doc, pred['diagnosis'], start_char=pred['start_char'], end_char=pred['end_char'])


				for ent in target_ents:
						results.append({
							"pmcid": int(doc['pmcid']),
							"diagnosis": resolved_target,
							"umls_cui": ent[0],
							"umls_name": ent[1],
							"umls_prob": ent[2],
							"count": pred['count'],
							"group_ix": ix,
							"start_char": pred['start_char'],
							"end_char": pred['end_char'],
						})

	return results

results = run_extraction(docs)

results_df = pd.DataFrame(results)

results_df.to_csv(output_dir / f'{base_name}_umls.csv', index=False)