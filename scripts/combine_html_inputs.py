""" Combine HTML files to a single DataFrame """

from pathlib import Path
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

html_files = Path('../data/html').glob('*/*.html')

# Load text, and pmid from filenames
data = []
for file in html_files:
    with open(file, 'r') as f:
        # Get pmid from filename
        html = f.read()
        body_text = BeautifulSoup(html, 'html.parser').get_text()
        pmid = file.stem

        complete = len(body_text) > 1000
        data.append((body_text, pmid, complete))

# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=['text', 'pmid', 'complete'])


def _convert_pmid_to_pmc(pmids):
    url_template = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids="

    # Chunk the PMIDs into groups of 200
    pmids = [str(p) for p in pmids]
    pmid_chunks = [pmids[i:i + 200] for i in range(0, len(pmids), 200)]

    pmc_ids = []
    for chunk in pmid_chunks:
        pmid_str = ','.join(chunk)
        url = url_template + pmid_str
        response = requests.get(url)
        # Respionse <record requested-id="23193288" pmcid="PMC3531191" pmid="23193288" doi="10.1093/nar/gks1163">
        pmc_ids += re.findall(r'<record requested-id="[^"]+" pmcid="([^"]+)" pmid="([^"]+)" doi="[^"]+">', response.text)

    pmids_found = set([p[1] for p in pmc_ids])
    missing_pmids = [(None, p) for p in pmids if p not in pmids_found]

    pmc_ids = pmc_ids + missing_pmids

    # Reverse the list of tuples, remove leading "PMC"
    pmc_ids = [(p, pmc) for pmc, p in pmc_ids]
    pmc_ids = [(p, pmc[3:]) for p, pmc in pmc_ids if pmc is not None]
    pmc_ids = dict(pmc_ids)

    return pmc_ids


# Convert PMIDs to PMCIDs
pmc_ids = _convert_pmid_to_pmc(df.pmid)
df['pmcid'] = df.pmid.map(pmc_ids)

df.to_csv('../data/html_combined.csv', index=False)
