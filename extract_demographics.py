import pandas as pd
import os
from pathlib import Path
from publang.pipelines import search_extract
from nipub_templates.prompts import FEW_SHOT_FC_2
from nipub_templates.clean import clean_predictions
from labelrepo.projects.participant_demographics import \
        get_participant_demographics
from labelrepo import database
from openai import OpenAI

# Load original annotations
subgroups = get_participant_demographics()
jerome_pd = subgroups[
    (subgroups.project_name == 'participant_demographics') &
    (subgroups.annotator_name == 'Jerome_Dockes')
    ]
subset_cols = [
    'count', 'diagnosis', 'group_name', 'subgroup_name', 'male count',
    'female count', 'age mean', 'age minimum', 'age maximum',
    'age median', 'pmcid']
jerome_pd_subset = jerome_pd[subset_cols].sort_values('pmcid')

# Load articles that have been annotated
docs = pd.read_sql(
    "select pmcid, text from document",
    database.get_database_connection(),
)
docs = docs[docs.pmcid.isin(jerome_pd.pmcid)].to_dict(orient='records')

output_dir = Path('outputs')

# Set up OpenAI clients
embed_model = 'text-embedding-ada-002'
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))
fireworks_client = OpenAI(api_key=os.getenv('FIREWORKS_API_KEY'), 
                          base_url='https://api.fireworks.ai/inference/v1')


def _run(model_name, extraction_client, min_chars, max_chars, prepend=''):
    prepend += '_'
    short_model_name = model_name.split('/')[-1]

    embeddings_path = output_dir / \
        f'eval_embeddings_minc-{min_chars}_maxc-{max_chars}.parquet'

    name = f"eval_{prepend}{short_model_name}_minc-{min_chars}_maxc-{max_chars}"
    predictions_path = output_dir / f'{name}.json'
    clean_predictions_path = output_dir / f'{name}_clean.csv'

    # Extract
    predictions = search_extract(
        articles=docs, output_path=predictions_path,
        min_chars=min_chars, max_chars=max_chars,
        embeds_path=embeddings_path, extraction_model=model_name,
        embed_model=embed_model, embed_client=openai_client,
        extraction_client=extraction_client,
        num_workers=1, **FEW_SHOT_FC_2
    )

    clean_predictions(predictions).to_csv(
        clean_predictions_path, index=False
    )


models = [
    ("accounts/fireworks/models/firefunction-v1", fireworks_client),
    # ("gpt-3.5-turbo-0613", openai_client),
    # ("gpt-4-0125-preview", openai_client),
]


# Split body into large sections (by setting min_chars to high number)
for model_name, client in models:
    _run(model_name, client, 40, 4000, prepend='demographics-few-shot')
