ZERO_SHOT_SAMPLE_SIZE_FUNCTION = {
    'messages': [
        {
            "role": "user", 
            "content":
                """You will be provided with a text sample from a scientific journal. 
                The sample is delimited with triple backticks.
                        
                Your task is to identify the final number of participants or patients that participated in the study, and underwent MRI.
                If you are not very confident that the number of participants is mentioned in the text sample, provide `null` as the response.

                Call the extractData function to save the output.

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

