""" Wrappers around OpenAI to make help embedding chunked documents """

import openai
import tqdm
from split import split_pmc_document

def embed_text(text, model_name='text-embedding-ada-002'):
    """ Embed a document using OpenAI's API """
    # Return the embedding
    response = openai.Embedding.create(
        input=text,
        model=model_name
    )
    embeddings = response['data'][0]['embedding']

    return embeddings


def embed_pmc_article(document, model_name='text-embedding-ada-002', min_tokens=20, max_tokens=4000):
    """ Embed a PMC article using OpenAI's API. 
    Split the article into chunks of min_tokens to max_tokens,
    and embed each chunk.
    """

    split_doc = split_pmc_document(document, min_tokens=min_tokens, max_tokens=max_tokens)

    # Embed each chunk
    for chunk in tqdm.tqdm(split_doc):
        res = embed_text(chunk['content'], model_name)
        chunk['embedding'] = res

    return split_doc
    
