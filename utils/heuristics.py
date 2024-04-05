import re
import pandas as pd

_PARTICIPANTS_SECTIONS = (
    r"(?:participants?|subjects?|patients|population|demographics?|design|procedure)"
)

_METHODS_SECTIONS = (
    r"methods?|materials?|design?|case|procedures?"
)

def get_chunks_heuristic(embeddings_df, section_2=True):
    """ if fallback=True, returns all if one step fails """
    results = []
    for _, sub_df in embeddings_df.groupby('pmcid', sort=False):
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
    