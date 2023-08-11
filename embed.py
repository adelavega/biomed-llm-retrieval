""" Wrappers around OpenAI to make help embedding chunked documents """

import openai
import tqdm
import pandas as pd
from split import split_pmc_document
from typing import Dict, List, Optional, Tuple

def embed_text(text: str, model_name: str = 'text-embedding-ada-002') -> List[float]:
    """ Embed a document using OpenAI's API """
    # Return the embedding
    response = openai.Embedding.create(
        input=text,
        model=model_name
    )
    embeddings = response['data'][0]['embedding']

    return embeddings


def embed_pmc_article(
        document: str, 
        model_name: str = 'text-embedding-ada-002', 
        min_tokens: int = 30, 
        max_tokens: int = 4000) -> List[Dict[str, any]]:
    """ Embed a PMC article using OpenAI's API. 
    Split the article into chunks of min_tokens to max_tokens,
    and embed each chunk.
    """

    split_doc = split_pmc_document(document, min_tokens=min_tokens, max_tokens=max_tokens)

    # Embed each chunk
    for chunk in split_doc:
        res = embed_text(chunk['content'], model_name)
        chunk['embedding'] = res

    return split_doc
    
def embed_pmc_articles(
        articles: List[Tuple], # [(pmcid, full_text), ...
        model_name: str = 'text-embedding-ada-002', 
        min_tokens: int = 30, 
        max_tokens: int = 4000) -> pd.DataFrame:
    """ Embed a list of PMC articles using OpenAI's API. 
    Split the article into chunks of min_tokens to max_tokens,
    and embed each chunk.
    """

    articles = []
    for pmcid, text in tqdm.tqdm(articles):
        results = embed_pmc_article(text, model_name, min_tokens, max_tokens)
        results['pmcid'] = pmcid
    return pd.concat(articles)