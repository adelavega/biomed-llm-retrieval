ZERO_SHOT_MULTI_GROUP_2 = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": """You will be provided with a text sample from a scientific journal. 
                The sample is delimited with triple backticks.
                        
                Your task is to identify groups of participants that participated in the study, and underwent MRI.

                If you are not confident in your answer or no sample size is mentioned, skip the question!
                You will be tipped $$500 for 90%+ percent accuracy.
                
                For each group identify:
                   - the number of participants in each group, and the diagnosis. 
                   - the number of male participants, and their mean age, median age, minimum and maximum age
                   - the number of female participants, and their mean age, median age, minimum and maximum age.

                If any of the information is missing, return `null` for that field.               
                
                Call the extractData function to save the output.
                Text sample: ${text}
                """,
        }
    ],
    "output_schema": {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "description": "Number of participants in this group",
                            "type": "integer",
                        },
                        "diagnosis": {
                            "description": "Diagnosis of the group, if any",
                            "type": "string",
                        },
                        "group_name": {
                            "description": "Group name, healthy or patients",
                            "type": "string",
                            "enum": ["healthy", "patients"],
                        },
                        "subgroup_name": {
                            "description": "Subgroup name",
                            "type": "string",
                        },
                        "male count": {
                            "description": "Number of male participants in this group",
                            "type": "integer",
                        },
                        "female count": {
                            "description": "Number of female participants in this group",
                            "type": "integer",
                        },
                        "age mean": {
                            "description": "Mean age of participants in this group",
                            "type": "number",
                        },
                        "age range": {
                            "description": "Age range of participants in this group, separated by a dash",
                            "type": "string",
                        },
                        "age minimum": {
                            "description": "Minimum age of participants in this group",
                            "type": "integer",
                        },
                        "age maximum": {
                            "description": "Maximum age of participants in this group",
                            "type": "integer",
                        },
                        "age median": {
                            "description": "Median age of participants in this group",
                            "type": "integer",
                        },
                    },
                    "required": ["count"],
                },
            }
        },
    },
}



ZERO_SHOT_MULTI_GROUP = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": """You will be provided with a text sample from a scientific journal. 
                The sample is delimited with triple backticks.
                        
                Your task is to identify groups of participants that participated in the study, and underwent MRI.
                
                For each group identify:
                   - the number of participants in each group, and the diagnosis. 
                   - the number of male participants, and their mean age, median age, minimum and maximum age
                   - the number of female participants, and their mean age, median age, minimum and maximum age.

                If any of the information is missing, return `null` for that field.               
                
                Call the extractData function to save the output.
                Text sample: ${text}
                """,
        }
    ],
    "output_schema": {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "description": "Number of participants in this group",
                            "type": "integer",
                        },
                        "diagnosis": {
                            "description": "Diagnosis of the group, if any",
                            "type": "string",
                        },
                        "group_name": {
                            "description": "Group name, healthy or patients",
                            "type": "string",
                            "enum": ["healthy", "patients"],
                        },
                        "subgroup_name": {
                            "description": "Subgroup name",
                            "type": "string",
                        },
                        "male count": {
                            "description": "Number of male participants in this group",
                            "type": "integer",
                        },
                        "female count": {
                            "description": "Number of female participants in this group",
                            "type": "integer",
                        },
                        "age mean": {
                            "description": "Mean age of participants in this group",
                            "type": "number",
                        },
                        "age range": {
                            "description": "Age range of participants in this group, separated by a dash",
                            "type": "string",
                        },
                        "age minimum": {
                            "description": "Minimum age of participants in this group",
                            "type": "integer",
                        },
                        "age maximum": {
                            "description": "Maximum age of participants in this group",
                            "type": "integer",
                        },
                        "age median": {
                            "description": "Median age of participants in this group",
                            "type": "integer",
                        },
                    },
                    "required": ["count"],
                },
            }
        },
    },
}

