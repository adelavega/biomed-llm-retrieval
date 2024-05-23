import spacy
from spacy import displacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector

import pandas as pd
from pathlib import Path
from labelrepo.projects.participant_demographics import \
        get_participant_demographics
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
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})


def match_umls(text, target):
	""" Given a text and a target, return the UMLS entities that match the target
	Takes advantage of abbreciation detection from full text and entity linking to UMLS.
	"""
	combined = f"{text} {target}"
	doc = nlp(combined)
	
	target_ents = []
	# Go backwards through doc entities to find the target
	for ent in reversed(doc.ents):
		if ent.start_char > len(text):
			target_ents.append(ent)
		else:
			break
	else:
		raise ValueError("Target not found in text")

	return target_ents

results = []
for doc in tqdm(docs):
	doc_preds = gpt_predictions[gpt_predictions.pmcid == doc['pmcid']]
	for _, pred in doc_preds.iterrows():
		# Get the UMLS entities that match the targettarg
		if pred['group_name'] == 'patients':
			target_ents = match_umls(doc['text'], pred['diagnosis'])
			for ent in target_ents:
				for cui, prob in ent._.kb_ents:
					results.append({
						"pmcid": int(doc['pmcid']),
						"diagnosis": pred['diagnosis'],
						"entity": ent.text,
						"umls_cui": cui,
						"umls_prob": prob,
						"count": pred['count']
					})

results_df = pd.DataFrame(results)
results_df.to_csv(output_dir / f"{base_name}_umls.csv", index=False)
