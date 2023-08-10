import re

def join_strings(strings, max_tokens=4000):
    """Join strings to form largest possible strings that are less than max_tokens."""
    chunks = []
    current_chunk = ''
    for string in strings:
        if len(current_chunk) + len(string) + 1 <= max_tokens:
            if current_chunk:
                current_chunk += '\n' + string
            else:
                current_chunk = string
        else:
            chunks.append(current_chunk)
            current_chunk = string
    chunks.append(current_chunk)
    return chunks


def split_markdown(text, delimiters, min_tokens=None, max_tokens=None):
    """Split markdown text into chunks based on delimiters.

    Args:
        text (str): Markdown text to split.
        delimiters (list): List of delimiters to split on.
        top_level (bool): Whether or not the current text is top level.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        list: List of chunks.
    """

    if not delimiters:
        # Join lines to form largest possible strings that are less than max_tokens
        return join_strings(text.splitlines(), max_tokens=max_tokens)

    # Split on first delimiter
    candidate_chunks = re.split(f'\n\s*{delimiters[0]}', text)

    # If there is only one chunk, split on next delimiter
    if len(candidate_chunks) == 1:
        return split_markdown(text, delimiters[1:], min_tokens, max_tokens)
    
    # Iterate over chunks
    chunks = []
    prev_chunk = None
    for ix, chunk in enumerate(candidate_chunks):
        if chunk:
            if not ix == 0:
                chunk = delimiters[0] + chunk
            if prev_chunk:
                chunk = prev_chunk + "\n" + chunk
                prev_chunk = None
            if min_tokens and len(chunk) < min_tokens:
                prev_chunk = chunk
                continue
            if max_tokens and len(chunk) > max_tokens:
                chunks += split_markdown(chunk, delimiters[1:], min_tokens, max_tokens)
            else:
                chunks.append(chunk)

    return chunks


def split_pmc_document(text, delimiters=['## ', '### '], min_tokens=20, max_tokens=4000):
    """Split PMC document text into chunks based on delimiters, and split by top level sections.

    Args:
        text (str): Markdown text to split.
        delimiters (list): List of delimiters to split on.
        min_tokens (int): Minimum number of tokens per chunk (for headers)
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        list: List of chunks.
    """

    sections = re.split(f'\n\s*# ', text)

    _outputs = {}

    for ix, section in enumerate(sections):
        if ix == 0:
            _outputs['Authors'] = section
        else:
            section_name, content = section.split('\n', maxsplit=1)
            _outputs[section_name] = content

    _outputs['Body'] = split_markdown(_outputs['Body'], delimiters, 
                                      min_tokens, max_tokens)

    return _outputs