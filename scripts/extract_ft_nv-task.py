""" Extract participant demographics from HTML files. """
import os
from publang.extract import extract_from_text
from openai import OpenAI
from pathlib import Path
import json
import pandas as pd

# Change directory for importing
import sys
sys.path.append('../')
from nipub_templates.nv_task.prompts import ZERO_SHOT_TASK


# Read JSON lines
docs = pd.read_json('../../labelbuddy-annotations/projects/nv_task/documents/task_0.jsonl', lines=True)

output_dir = Path('../outputs/nv_task/extractions')
output_dir.mkdir(parents=True, exist_ok=True)

# Set up OpenAI clients
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))


def _run(extraction_model, extraction_client, docs, prepend='', **extract_kwargs):
    prepend += '_'
    short_model_name = extraction_model.split('/')[-1]
    
    pmcids = [d['pmcid'] for d in docs['metadata']]

    name = f"full_{prepend}{short_model_name}"
    predictions_path = output_dir / f'{name}.json'

    # Extract
    predictions = extract_from_text(
        docs['text'].to_list(),
        model=extraction_model, client=extraction_client,
        **extract_kwargs
    )

    # Add abstract id to predictions
    outputs = []
    for pred, _id in zip(predictions, pmcids):
        if pred:    
            pred['pmcid'] = _id
            outputs.append(pred)

    json.dump(outputs, open(predictions_path, 'w'))


models = [
    ("gpt-4o-2024-08-06", openai_client, {"temperature": 1}),
    # ("anthropic/claude-3.5-sonnet", openrouter_client),
    # ("gpt-4o-2024-05-13", openai_client)
]

for model_name, client, kwargs in models:
    _run(model_name, client, docs, prepend='lb_nv_taskstructured-zeroshot',
         **ZERO_SHOT_TASK, num_workers=10, **kwargs)
