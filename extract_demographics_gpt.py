import pandas as pd
import os
from labelrepo.projects.participant_demographics import get_participant_demographics
from labelrepo.database import get_database_connection
from publang.pipelines import extract_gpt_demographics

api_key = open('/home/zorro/.keys/open_ai.key').read().strip()

# Load annotations
subgroups = get_participant_demographics()
jerome_pd = subgroups[(subgroups.project_name == 'participant_demographics') & \
                      (subgroups.annotator_name == 'Jerome_Dockes')]
subset_cols = ['count', 'diagnosis', 'group_name', 'subgroup_name', 'male count',
       'female count', 'age mean', 'age minimum', 'age maximum',
       'age median', 'pmcid']
jerome_pd_subset = jerome_pd[subset_cols].sort_values('pmcid')

# Load articles
docs = pd.read_sql(
    "select pmcid, text from document",
    get_database_connection(),
)
docs = docs[docs.pmcid.isin(jerome_pd.pmcid)].to_dict(orient='records')

max_tokens = 1000

predictions_path = f'data/participant_demographics_gpt_maxtokens-{max_tokens}.csv'
embeddings_path = f'data/evaluation_embeddings_maxtokens-{max_tokens}.csv'

# Try to open each, if not set to None
embeddings, predictions = None, None
if os.path.exists(embeddings_path):
    embeddings = pd.read_csv(embeddings_path)
if os.path.exists(predictions_path):
    predictions = pd.read_csv(predictions_path)

if predictions is None:
    # Extract
    predictions, embeddings = extract_gpt_demographics(
        articles=docs, embeddings=embeddings, api_key=api_key, max_tokens=max_tokens, num_workers=10
    )
    # Save predictions
    predictions.to_csv(predictions_path, index=False)
    embeddings.to_csv(embeddings_path, index=False)
