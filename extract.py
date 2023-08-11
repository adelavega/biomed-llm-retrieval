""" Provides a high-level API for LLMs for the purpose of infomation retrieval from documents and evaluation of the results."""

import openai
import json
import re 
import pandas as pd
import tqdm 
import warnings 
import concurrent.futures
import requests
import time

from typing import List, Dict, Union

def get_openai_completion(
        prompt: str, 
        model_name: str, 
        temperature: float = 0, 
        request_timeout: int = 3, 
        num_retries: int = 3) -> str:
    # Check that model_name is a valid model name before using it in the openai.ChatCompletion.create() call.
    valid_models = ["gpt-3.5-turbo", "gpt-4"]
    if model_name not in valid_models:
        raise ValueError(f"{model_name} is not a valid OpenAI model name.")

    for i in range(num_retries):
        try:
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                request_timeout=request_timeout
            )

        except requests.exceptions.HTTPError as e:
            if isinstance(e.__cause__, requests.exceptions.HTTPError) and e.__cause__.response.status_code == 429:
                # If we get a 429 error, wait for the specified amount of time and retry
                retry_after = int(e.__cause__.response.headers.get("Retry-After", "5"))
                print(f"Received 429 error. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                # If it's not a 429 error, re-raise the exception
                raise e
        except requests.exceptions.RequestException as e:
            # If we get any other kind of exception, retry after a short delay
            print(f"Received {type(e).__name__} exception. Retrying after 1 second...")
            time.sleep(1)
        except openai.error.RateLimitError as e:
            # If we get a RateLimitError, retry after a short delay
            print(f"Received {type(e).__name__} exception. Retrying after 1 minute...")
            time.sleep(61)


    if not completion['choices'] or not completion['choices'][0]['message'].get('content'):
        raise ValueError("The completion must contain a 'choices' list with at least one element, \
                         and the first element must contain a 'message' dictionary with a non-empty 'content' field.")
    
    message = completion['choices'][0]['message']['content']

    return message


def format_string_with_variables(string: str, **kwargs: str) -> str:
    # Find all possible variables in the string
    possible_variables = set(re.findall(r"{(\w+)}", string))

    # Find all provided variables in the kwargs dictionary
    provided_variables = set(kwargs.keys())

    # Check that all possible variables are provided
    if possible_variables != provided_variables:
        raise ValueError(f"Not all possible variables are provided. Missing variables: {possible_variables - provided_variables}")

    # Format the string with the provided variables
    return string.format(**kwargs)


def validate_completion_keys(completion: dict, expected_keys: List[str]) -> bool:
    """Checks that the completion contains all the expected keys."""
    if not expected_keys:
        return
    completion_keys = set(completion.keys())
    if completion_keys != set(expected_keys):
        raise ValueError(f"Completion keys {completion_keys} do not match the expected keys {expected_keys}.")
    

def extract_from_text(
        text: str, 
        template: str, 
        expected_keys: List[str] = None, 
        model_name: str = "gpt-3.5-turbo") -> Dict[str, str]:
    """Extracts information from a text sample using an OpenAI LLM.

    Args:
        text: A string containing the text sample.
        template: A dictionary containing the template for the prompt and the expected keys in the completion.
        model_name: A string containing the name of the LLM to be used for the extraction.
    """

    # Encode text to ascii
    text = text.encode("ascii", "ignore").decode()

    prompt = format_string_with_variables(template, text=text)
    completion = get_openai_completion(prompt, model_name)

    try:
        data = json.loads(completion)
    except json.decoder.JSONDecodeError:
        warnings.warn("Completion is not a valid JSON. Returning the completion as a string.")
        data = completion

    validate_completion_keys(data, expected_keys)

    return data

def extract_from_multiple(
        texts: List[str], 
        template: str, 
        expected_keys: List[str] = None, 
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
                executor.submit(extract_from_text, text, template, expected_keys, model_name) 
                for text in texts
                ]

            results = []
            for future in tqdm.tqdm(futures, total=len(texts)):
                results.append(future.result())

    else:
        results = []
        for text in tqdm.tqdm(texts):
            results.append(
                extract_from_text(text, template, expected_keys, model_name))

    if return_type == "pandas":
        results = pd.DataFrame(results)

    return results