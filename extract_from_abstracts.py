""" Script to extract participant demographic information from abstracts of studies used in neuroimaging meta-analyses """

import pandas as pd
import os
import json
from pathlib import Path
from publang.extract import extract_from_text
from nipub_templates.clean import clean_predictions
from nipub_templates.demographics_orig import ZERO_SHOT_MULTI_GROUP
from openai import OpenAI


output_dir = Path('outputs')

# Set up OpenAI clients
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))


docs = pd.read_csv('data/kendra_abstracts.csv')
docs = docs[docs.abstract.isna() == False]


def _run(model_name, extraction_client, prepend='', **kwargs):
    prepend += '_'
    short_model_name = model_name.split('/')[-1]

    kwargs.pop('search_query', None)

    name = f"metaabstracts_{prepend}{short_model_name}"
    predictions_path = output_dir / f'{name}.json'
    clean_predictions_path = output_dir / f'{name}_clean.csv'


    # Extract
    predictions = extract_from_text(
        docs['abstract'].to_list(),
        model=model_name, client=extraction_client,
        num_workers=10,
        **kwargs
    )

    # Add abstract id to predictions
    outputs = []
    for pred, _id in zip(predictions, docs['id']):
        if pred:
            pred['id'] = _id
            outputs.append(pred)

    json.dump(outputs, open(predictions_path, 'w'))

    clean_predictions(outputs).to_csv(
        clean_predictions_path, index=False
    )


models = [
    ("gpt-3.5-turbo-0613", openai_client)
]


# Split body into large sections (by setting min_chars to high number)
# Running this using the original prompt from the lit mining paper.
for model_name, client in models:
    _run(model_name, client, prepend='demographics-abstracts-2', 
         **ZERO_SHOT_MULTI_GROUP)
