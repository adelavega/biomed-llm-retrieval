""" Provides a high-level API for LLMs for the purpose of infomation retrieval from documents and evaluation of the results."""

import openai
import json
import re 
import pandas as pd
import tqdm 
import concurrent.futures
from embed import get_chunk_query_distance
from copy import deepcopy

from typing import List, Dict, Union

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

@retry(
    retry=retry_if_exception_type((
        openai.error.APIError, 
        openai.error.APIConnectionError, 
        openai.error.RateLimitError, 
        openai.error.ServiceUnavailableError, 
        openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def format_function(parameters):
    """ Format function for OpenAI function calling from parameters"""
    functions = [
        {
            'name': 'extractData',
            'parameters': parameters
        }

    ]

    function_call = {"name": "extractData"}

    return functions, function_call

def get_openai_json_response(
        messages: List[Dict[str, str]],
        parameters: Dict[str, object],
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0, 
        request_timeout: int = 10) -> str:
    # Check that model_name is a valid model name before using it in the openai.ChatCompletion.create() call.
    valid_models = ["gpt-3.5-turbo", "gpt-4"]
    if model_name not in valid_models:
        raise ValueError(f"{model_name} is not a valid OpenAI model name.")
    
    functions, function_call = format_function(parameters)

    completion = chat_completion_with_backoff(
        model=model_name,
        messages=messages,
        functions=functions,
        function_call=function_call,
        temperature=temperature,
        request_timeout=request_timeout
    )
    
    c = completion['choices'][0]['message']
    res = c['function_call']['arguments']
    data = json.loads(res)

    return data


def format_string_with_variables(string: str, **kwargs: str) -> str:
    # Find all possible variables in the string
    possible_variables = set(re.findall(r"{(\w+)}", string))

    # Find all provided variables in the kwargs dictionary
    provided_variables = set(kwargs.keys())

    # Check that all provided variables are in the possible variables
    if not provided_variables.issubset(possible_variables):
        raise ValueError(f"Provided variables {provided_variables} are not in the possible variables {possible_variables}.")

    # Format the string with the provided variables
    return string.format(**kwargs)

def extract_from_text(
        text: str, 
        messages: str,
        parameters: Dict[str, object],
        model_name: str = "gpt-3.5-turbo") -> Dict[str, str]:
    """Extracts information from a text sample using an OpenAI LLM.

    Args:
        text: A string containing the text sample.
        template: A dictionary containing the template for the prompt and the expected keys in the completion.
        model_name: A string containing the name of the LLM to be used for the extraction.
    """

    # Encode text to ascii
    text = text.encode("ascii", "ignore").decode()

    messages = deepcopy(messages)

    # Format the message with the text
    for message in messages:
        message['content'] = format_string_with_variables(message['content'], text=text)

    data = get_openai_json_response(
        messages,
        parameters=parameters,
        model_name=model_name
    )

    return data

def extract_from_multiple(
        texts: List[str], 
        messages: str,
        parameters: Dict[str, object],
        model_name: str = "gpt-3.5-turbo", 
        num_workers: int = 1) -> Union[List[Dict[str, str]], pd.DataFrame]:
    """Extracts information from multiple text samples using an OpenAI LLM.

    Args:
        texts: A list of strings containing the text samples.
        template: A dictionary containing the template for the prompt and the expected keys in the completion.
        model_name: A string containing the name of the LLM to be used for the extraction.
        return_type: A string specifying the type of the returned object. Can be either "pandas" or "list".
        num_workers: An integer specifying the number of workers to use for parallel processing.
    """

    if num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    extract_from_text, text, messages, parameters, model_name) 
                for text in texts
                ]

            results = []
            for future in tqdm.tqdm(futures, total=len(texts)):
                results.append(future.result())

    else:
        results = []
        for text in tqdm.tqdm(texts):
            results.append(
                extract_from_text(text, messages, parameters, model_name))

    return results

def get_relevant_chunks(embeddings_df, annotations_df):
    """ Get relevant chunks from embeddings_df based on annotations_df"""
    # Find first chunk that contains a true annotation
    sections = []
    # For every section, see if it contains any annotations
    for ix, row in embeddings_df.iterrows():
        annotations  = annotations_df[annotations_df.pmcid == row['pmcid']]
        for ix_a, annot in annotations.iterrows():
            contains = [True for s, e in  zip(annot['start_char'], annot['end_char']) if (row.start_char <= s) & (row.end_char >= e)]
            if any(contains):
                sections.append(ix)
                break
    return embeddings_df.loc[sections]

_PARTICIPANTS_SECTIONS = (
    r"(?:participants?|subjects?|patients|population|demographics?|design|procedure)"
)

_METHODS_SECTIONS = (
    r"methods?|materials?|design?|case|procedures?"
)

def get_chunks_heuristic(embeddings_df, section_2=True):
    """ if fallback=True, returns all if one step fails """
    results = []
    for pmcid, sub_df in embeddings_df.groupby('pmcid', sort=False):
        # Try getting Methods section
        m_ix = sub_df.section_1.apply(
            lambda x: bool(re.search(_METHODS_SECTIONS, x,  re.IGNORECASE)) if not pd.isna(x) else False)

        if m_ix.sum() != 0:
            sub_df = sub_df[m_ix]

            # Try getting design section
            d_ix = sub_df.section_2.apply(
                lambda x: bool(re.search(_PARTICIPANTS_SECTIONS, x,  re.IGNORECASE)) if not pd.isna(x) else False)
            if section_2 and d_ix.sum() > 0:
                sub_df = sub_df[d_ix]
        results.append(sub_df)
    
    return pd.concat(results)
    
def extract_on_match(
        embeddings_df, annotations_df, messages, parameters, model_name="gpt-3.5-turbo", 
        num_workers=1):
    """ Extract anntotations on chunk with relevant information (based on annotation meta data) """

    embeddings_df = embeddings_df[embeddings_df.section_0 == 'Body']

    sections = get_relevant_chunks(embeddings_df, annotations_df)

    res = extract_from_multiple(sections.content.to_list(), messages, parameters, 
                          model_name=model_name, num_workers=num_workers)

    # Combine results into single df and add pmcid
    pred_groups_df = []
    for ix, r in enumerate(res):
        rows = r['groups']
        pmcid = sections.iloc[ix]['pmcid']
        for row in rows:
            row['pmcid'] = pmcid
            pred_groups_df.append(row)
    pred_groups_df = pd.DataFrame(pred_groups_df)

    return sections, pred_groups_df


def _extract_iteratively(
        sub_df, messages, parameters, model_name="gpt-3.5-turbo"):
    """ Iteratively attempt to extract annotations from chunks in ranks_df until one succeeds. """
    for _, row in sub_df.iterrows():
        res = extract_from_text(row['content'], messages, parameters, model_name)
        if res['groups']:
            result = [
                {**r, **row[['rank', 'start_char', 'end_char', 'pmcid']].to_dict()} for r in res['groups']
                ]
            return result
        
    return []
    

def search_extract(
        embeddings_df, query, messages, parameters, model_name="gpt-3.5-turbo", 
        heuristic_strategy=None, num_workers=1):
    """ Search for query in embeddings_df and extract annotations from nearest chunks,
    using heuristic to narrow down search space if specified.
    """

    # Use heuristic to narrow down search space
    if heuristic_strategy == None:
        embeddings_df = embeddings_df[embeddings_df.section_0 == 'Body']
    elif heuristic_strategy == 'methods':
        embeddings_df = get_chunks_heuristic(embeddings_df, section_2=False)
    elif heuristic_strategy == 'demographics':
        embeddings_df = get_chunks_heuristic(embeddings_df, section_2=True)

    # Search for query in chunks
    ranks_df = get_chunk_query_distance(embeddings_df, query)
    ranks_df.sort_values('distance', inplace=True)
    
    # For every document, try to extract annotations by distance until one succeeds
    if num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _extract_iteratively, sub_df, messages, parameters, model_name) 
                for _, sub_df in ranks_df.groupby('pmcid', sort=False)
                ]

            results = []
            for future in tqdm.tqdm(futures, total=len(ranks_df.pmcid.unique())):
                results.extend(future.result())
    else:
        results = []
        for _, sub_df in tqdm.tqdm(ranks_df.groupby('pmcid', sort=False)):
            results.extend(_extract_iteratively(sub_df, messages, parameters, model_name))

    results = pd.DataFrame(results)

    return results