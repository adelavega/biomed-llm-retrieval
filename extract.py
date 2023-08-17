""" Provides a high-level API for LLMs for the purpose of infomation retrieval from documents and evaluation of the results."""

import openai
import json
import re 
import pandas as pd
import tqdm 
import concurrent.futures
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
        return_type: str = "pandas", 
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

    if return_type == "pandas":
        results = pd.DataFrame(results)

    return results