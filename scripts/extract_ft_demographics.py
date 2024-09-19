""" Extract participant demographics from HTML files. """
import os
import pandas as pd
from publang.pipelines import search_extract
from publang.extract import extract_from_text
from openai import OpenAI
from pathlib import Path
from labelrepo.projects.participant_demographics import \
        get_participant_demographics
from labelrepo import database
import json

# Change directory for importing
import sys
sys.path.append('../')
from nipub_templates.prompts import ZERO_SHOT_MULTI_GROUP_FC, ZERO_SHOT_MULTI_GROUP_FTSTRICT_FC
from nipub_templates.clean import clean_predictions


html_docs = pd.read_csv('../data/html_combined.csv')
html_docs = html_docs[html_docs.complete == True]

output_dir = Path('../outputs/extractions')

# Set up OpenAI clients
embed_model = 'text-embedding-ada-002'
# openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))
# fireworks_client = OpenAI(api_key=os.getenv('FIREWORKS_API_KEY'), 
                        #   base_url='https://api.fireworks.ai/inference/v1')
openrouter_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Change directory for importing
import sys
sys.path.append('../')

# Load original annotations
combined_annotations = pd.read_csv('../annotations/combined_pd.csv')

# Load articles that have been annotated
md_docs = pd.read_sql(
    "select pmcid, text from document",
    database.get_database_connection(),
)

md_docs = md_docs[
    md_docs.pmcid.isin(combined_annotations.pmcid)]


def _run(extraction_model, extraction_client, docs, prepend='', **extract_kwargs):
    prepend += '_'
    short_model_name = extraction_model.split('/')[-1]

    extract_kwargs.pop('search_query', None)

    name = f"full_{prepend}{short_model_name}"
    predictions_path = output_dir / f'{name}.json'
    clean_predictions_path = output_dir / f'{name}_clean.csv'

    # Extract
    predictions = extract_from_text(
        docs['text'].to_list(),
        model=extraction_model, client=extraction_client,
        **extract_kwargs
    )

    # Add abstract id to predictions
    outputs = []
    for pred, _id in zip(predictions, docs['pmcid']):
        if pred:    
            pred['pmcid'] = _id
            outputs.append(pred)

    json.dump(outputs, open(predictions_path, 'w'))

    clean_predictions(predictions).to_csv(
        clean_predictions_path, index=False
    )


models = [
    # ("gpt-4o-mini-2024-07-18", openai_client),
    ("anthropic/claude-3.5-sonnet", openrouter_client),
    # ("gpt-4o-2024-05-13", openai_client)
]


# for model_name, client in models:
#     _run(model_name, client, html_docs, prepend='html_demographics-zeroshot',
#          **ZERO_SHOT_MULTI_GROUP_FC, num_workers=10)

for model_name, client in models:
    _run(model_name, client, md_docs, prepend='md_demographics-zeroshot',
         **ZERO_SHOT_MULTI_GROUP_FC, num_workers=10)
