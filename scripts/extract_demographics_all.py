import pandas as pd
import os
from pathlib import Path
from nipub_templates.prompts import ZERO_SHOT_MULTI_GROUP_FC
from nipub_templates.clean import clean_predictions
from publang.pipelines import search_extract
from openai import OpenAI


inputs = pd.read_csv('/data/alejandro/projects/ns-pond/source/mega-ni-dataset/pubget_searches/fmri_journal/query_875641cf4cbc22f32027447cd62fca27/subset_allArticles_extractedData/text.csv')


output_dir = Path('../outputs/extractions')

# Set up OpenAI clients
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))
fireworks_client = OpenAI(api_key=os.getenv('FIREWORKS_API_KEY'), 
                          base_url='https://api.fireworks.ai/inference/v1')


def _run(extraction_model, extraction_client, min_chars, max_chars,
         prepend='', **extract_kwargs):
    prepend += '_'
    short_model_name = extraction_model.split('/')[-1]

    embeddings_path = output_dir / \
        f'all_embeddings_minc-{min_chars}_maxc-{max_chars}.parquet'

    name = f"all_{prepend}{short_model_name}_minc-{min_chars}_maxc-{max_chars}"
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
    # ("accounts/fireworks/models/firefunction-v1", fireworks_client),
    ("gpt-4o-2024-05-13", openai_client),
]


# Split body into large sections (by setting min_chars to high number)
for model_name, client in models:
    _run(model_name, client, 40, 4000, prepend='demographics-zeroshot',
         **ZERO_SHOT_MULTI_GROUP_FC, num_workers=10)