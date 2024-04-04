import pandas as pd
from pathlib import Path
from publang_pipelines.demographics import extract_gpt_demographics, clean_gpt_demo_predictions
from labelrepo.projects.participant_demographics import get_participant_demographics
from labelrepo import database


### Training documents
# Load original annotations
subgroups = get_participant_demographics()
jerome_pd = subgroups[(subgroups.project_name == 'participant_demographics') & \
                      (subgroups.annotator_name == 'Jerome_Dockes')]
subset_cols = ['count', 'diagnosis', 'group_name', 'subgroup_name', 'male count',
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

model_name = 'gpt-3.5-turbo-1106'

output_dir = Path('outputs')

docs = docs[0:2]

for max_chars in [10000]:

    predictions_path = output_dir / f'eval_participant_demographics_{model_name}_tokens-{max_chars}.csv'
    clean_predictions_path = output_dir / f'eval_participant_demographics_{model_name}_tokens-{max_chars}_clean.csv'
    embeddings_path = output_dir / f'eval_embeddings_tokens-{max_chars}.parquet'

    # Extract
    predictions = extract_gpt_demographics(
        articles=docs, output_path=predictions_path, max_chars=max_chars, num_workers=1,
        embeddings_path=embeddings_path, extraction_model_name=model_name
    )

    clean_gpt_demo_predictions(predictions).to_csv(clean_predictions_path, index=False)
    predictions.to_csv(predictions_path, index=False)