import spacy
from spacy import displacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector


nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

doc = nlp(body)


for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")


linker = nlp.get_pipe("scispacy_linker")
for umls_ent in doc.ents[2]._.kb_ents:
	print(linker.kb.cui_to_entity[umls_ent[0]])