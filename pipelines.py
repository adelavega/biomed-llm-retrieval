from typing import List, Dict
import pandas as pd
import os
import openai
from embed import embed_pmc_articles
from extract import search_extract
from templates import ZERO_SHOT_MULTI_GROUP
from evaluate import clean_predictions

def extract_gpt_demographics(
        articles: List[Dict] = None,
        embeddings: pd.DataFrame = None,
        api_key: str = None,
        embedding_model_name: str = 'text-embedding-ada-002', 
        min_tokens: int = 30, 
        max_tokens: int = 4000,
        query: str = None,
        heuristic_strategy: str = None,
        template: dict = None,
        extraction_model_name: str = 'gpt-3.5-turbo',
        clean_preds: bool = True,
        num_workers: int = 1
        ) -> (pd.DataFrame, pd.DataFrame):
    """Extract participant demographics from a list of articles using OpenAI's API.

    Args:
        articles (list): List of articles. Each article is a dictionary with keys 'pmcid' and 'text'.
        embeddings (pd.DataFrame): Embeddings for each article. If None, embeddings will be computed.
                                   if provided, articles are not required.
        api_key (str): OpenAI API key. If None, the key will be read from the OPENAI_API_KEY environment variable.
        embedding_model_name (str): Name of the OpenAI embedding model to use.
        min_tokens (int): Minimum number of tokens per chunk.
        max_tokens (int): Maximum number of tokens per chunk.
        query (str): Query to use for the extraction.
        heuristic_strategy (str): Heuristic strategy to use for the extraction.
        template (dict): Template to use for the extraction (see templates.py).
        extraction_model_name (str): Name of the OpenAI model to use for the extraction.
        clean_preds (bool): Whether to clean the predictions.
        num_workers (int): Number of workers to use for parallel processing.

    Returns:
        pd.DataFrame: Dataframe containing the extracted values.
        pd.DataFrame: Dataframe containing the chunked embeddings for each article.
    """

    if api_key is not None:
        openai.api_key = api_key
    else:
        openai.api_key = os.environ.get('OPENAI_API_KEY', None)

    if embeddings is None:
        if articles is None:
            raise ValueError('Either articles or embeddings must be provided.')
        print('Embedding articles...')
        embeddings = embed_pmc_articles(articles, embedding_model_name, min_tokens, max_tokens)
        embeddings = pd.DataFrame(embeddings)

    if query is None:
        query = 'How many participants or subjects were recruited for this study?' 

    if template is None:
        template = ZERO_SHOT_MULTI_GROUP

    print('Extracting demographics...')
    predictions = search_extract(
        embeddings, query, **template, 
        heuristic_strategy=heuristic_strategy, 
        model_name=extraction_model_name, 
        num_workers=num_workers
        )
    
    if clean_preds:
        predictions = clean_predictions(predictions)

    return predictions, embeddings