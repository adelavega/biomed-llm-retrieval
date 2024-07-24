import pandas as pd
import os
from pathlib import Path
from publang.pipelines import search_extract
from labelrepo.projects.participant_demographics import \
        get_participant_demographics
from labelrepo import database
from openai import OpenAI

# Change directory for importing
import sys
sys.path.append('../')

from nipub_templates.prompts import ZERO_SHOT_MULTI_GROUP_FC, FEW_SHOT_FC
from nipub_templates.clean import clean_predictions

# Load original annotations
combined_annotations = pd.read_csv('../annotations/combined_pd.csv')

# Load articles that have been annotated
docs = pd.read_sql(
    "select pmcid, text from document",
    database.get_database_connection(),
)

docs = docs[
    docs.pmcid.isin(combined_annotations.pmcid)].to_dict(orient='records')

output_dir = Path('../outputs')

# Set up OpenAI clients
embed_model = 'text-embedding-ada-002'
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))
fireworks_client = OpenAI(api_key=os.getenv('FIREWORKS_API_KEY'), 
                          base_url='https://api.fireworks.ai/inference/v1')
openrouter_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)


def _run(extraction_model, extraction_client, min_chars, max_chars,
         prepend='', **extract_kwargs):
    prepend += '_'
    short_model_name = extraction_model.split('/')[-1]

    embeddings_path = output_dir / \
        f'chunked_embeddings_minc-{min_chars}_maxc-{max_chars}.parquet'

    name = f"chunked_{prepend}{short_model_name}_minc-{min_chars}_maxc-{max_chars}"
    predictions_path = output_dir / f'{name}.json'
    clean_predictions_path = output_dir / f'{name}_clean.csv'

    # Extract
    predictions = search_extract(
        articles=docs, output_path=predictions_path,
        min_chars=min_chars, max_chars=max_chars,
        embeds_path=embeddings_path, extraction_model=extraction_model,
        embed_model=embed_model, embed_client=openai_client,
        extraction_client=extraction_client,
        **extract_kwargs
    )

    clean_predictions(predictions).to_csv(
        clean_predictions_path, index=False
    )


models = [
    # ("anthropic/claude-3-haiku", openrouter_client),
    # ("anthropic/claude-3.5-sonnet", openrouter_client),
    # ("accounts/fireworks/models/firefunction-v2", fireworks_client),
    # ("gpt-3.5-turbo-0613", openai_client),
    # ("gpt-4-0125-preview", openai_client),
    ("gpt-4o-mini-2024-07-18", openai_client),
]


# Split body into large sections (by setting min_chars to high number)
for model_name, client in models:
    _run(model_name, client, 40, 4000, prepend='demographics-fewshot',
         **FEW_SHOT_FC, num_workers=10)