import re

def split_lines(text, max_tokens=100):
    """Join strings to form largest possible strings that are less than max_tokens."""

    strings = text.splitlines()
    if text[-1] == '\n':
        strings[-1] = strings[-1] + '\n'

    chunks = []
    current_chunk = ''
    for ix, string in enumerate(strings):
        if ix != 0:
            string = '\n' + string
        if len(current_chunk) + len(string) + 1 <= max_tokens:
                current_chunk += string
        else:
            if current_chunk != '':
                chunks.append(current_chunk)
            current_chunk = string
    chunks.append(current_chunk)

    if strings[-1] == '':
        chunks[-1] = chunks[-1] + '\n'

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
        return split_lines(text, max_tokens=max_tokens)
    
    delim_match = f'\n{delimiters[0]}'

    # Split on first delimiter
    candidate_chunks = re.split(delim_match, text)

    # If there is only one chunk, split on next delimiter
    if len(candidate_chunks) == 1:
        return split_markdown(text, delimiters[1:], min_tokens, max_tokens)
    
    # Iterate over chunks
    chunks = []
    prev_chunk = None
    for ix, chunk in enumerate(candidate_chunks):
        if chunk:
            if not ix == 0:
                chunk = delim_match + chunk
            if prev_chunk:
                chunk = prev_chunk + chunk
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

    sections = re.split(f'\n# ', text)

    _outputs = []

    start_char = 0
    chunk_id = 0
    for ix, content in enumerate(sections):
        if ix == 0:
            section_name = 'Authors'
        else:
            content = '\n# ' + content
            section_name, _ = content.replace('\n# ', '').split('\n', maxsplit=1)
            
        if section_name == 'Body':
            content = split_markdown(content, delimiters, min_tokens, max_tokens)
        else:
            content = [content]
            
        for ix, chunk in enumerate(content):
            end_char = start_char + len(chunk)
            _outputs.append({'section_name': section_name, 'content': chunk, 'chunk_id': ix, 'start_char': start_char, 'end_char': end_char})
            chunk_id += 1
            start_char = end_char
        
    return _outputs