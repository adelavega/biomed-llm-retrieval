ZERO_SHOT_SINGLE_GROUP_DEMOGRAPHICS = {
    'template': """
                You will be provided with a text sample from a scientific journal. 
                The sample is delimited with triple backticks.

                Perform the following tasks:
                1. Identify the total number of participants that underwent fMRI or neuroimaging in the study, if any. 
                2. Provide your response in a JSON format containing a single key `count` and a integer value corresponding to the number of participants. 
                Do not provide any additional information except the JSON. If the number of participants is not mentioned in the text, provide `n/a` as the value. If the number of participants is 0 return `n/a`.

                Text sample: ```{text}```

                Your JSON response:
                """,
    'expected_keys': ['count']
}


