ZERO_SHOT_SAMPLE_SIZE_FUNCTION = {
    'messages': [
        {
            "role": "user", 
            "content":
                """You will be provided with a text sample from a scientific journal. 
                The sample is delimited with triple backticks.
                        
                Your task is to identify the total number of participants that underwent fMRI or neuroimaging in the study, if any. 
                If the number of participants is not mentioned in the text, provide null as the value.

                Call the extractData function to save the participant count. 

                Text sample: ```{text}```
                """
        }
    ],
    'parameters':
        {
            'type': 'object',
            'properties': {
                'count': {
                    'title': 'Count', 
                    'type': 'integer'
                    },
            }, 
            'required': ['count']
        }
}

