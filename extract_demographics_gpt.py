import pandas as pd
from pathlib import Path
from publang.pipelines import search_extract
from nipub_templates.demographics import (
    clean_gpt_demo_predictions, ZERO_SHOT_MULTI_GROUP)
from labelrepo.projects.participant_demographics import \
        get_participant_demographics
from labelrepo import database

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

# database.make_database()

# Load articles
docs = pd.read_sql(
    "select pmcid, text from document",
    database.get_database_connection(),
)
docs = docs[docs.pmcid.isin(jerome_pd.pmcid)].to_dict(orient='records')

# Test
docs = docs[:10]

model_name = 'gpt-3.5-turbo-0613'
output_dir = Path('outputs')
min_chars = 40


def _run(model_name, min_chars, max_chars):
    predictions_path = output_dir / \
        f'eval_participant_demographics_{model_name}_minchars-{min_chars}_maxchars-{max_chars}_test.json'
    clean_predictions_path = output_dir / \
        f'eval_participant_demographics_{model_name}_minchars-{min_chars}_maxchars-{max_chars}_clean_test.csv'
    embeddings_path = output_dir / \
        f'eval_embeddings_minchars-{min_chars}_maxchars-{max_chars}_test.parquet'

    # Extract
    predictions = search_extract(
        articles=docs, output_path=predictions_path,
        min_chars=min_chars, max_chars=max_chars,
        embeds_path=embeddings_path, extraction_model=model_name,
        num_workers=6, **ZERO_SHOT_MULTI_GROUP
    )

    clean_gpt_demo_predictions(predictions).to_csv(
        clean_predictions_path, index=False
    )


# Extract using chunks
for model_name in ['gpt-4-0125-preview']:
    _run(model_name, min_chars, 4000)

# # Split body into large sections (by setting min_chars to high number)
# for model_name in ['gpt-3.5-turbo-1106', 'gpt-4-0125-preview']:
#     _run(model_name, 2000, 30000)

# Extract using full Body
# predictions_path = output_dir / \
#     f'eval_participant_demographics_{model_name}_tokens-all_body.json'
# clean_predictions_path = output_dir / \
#     f'eval_participant_demographics_{model_name}_tokens-all_body_clean.csv'

# # Get text body
# all_preds = []
# for doc in tqdm(docs):
#     split_doc = split_pmc_document(doc['text'], delimiters=['# '])
#     if split_doc is None:
#         continue

#     text = [section['content'] for section in split_doc
#             if section.get('section_0') == 'Body'][0]

#     prediction = extract_from_text(
#         text, model=model_name,
#         num_workers=6, messages=ZERO_SHOT_MULTI_GROUP['messages'],
#         output_schema=ZERO_SHOT_MULTI_GROUP['output_schema']
#     )

#     prediction['pmcid'] = doc['pmcid']
#     all_preds.append(prediction)

# json.dump(all_preds, open(predictions_path, 'w'))

# clean_gpt_demo_predictions(all_preds).to_csv(
#     clean_predictions_path, index=False)
