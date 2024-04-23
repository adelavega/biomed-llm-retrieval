import pandas as pd
import numpy as np

base_message = """You will be provided with a text sample from a scientific journal. 
                The sample is delimited with triple backticks.

                Your task is to identify groups of participants that participated in the study, and underwent MRI.
                If there is no mention of any participant groups, return a null array.

                For each group identify:
                   - the number of participants in each group, and the diagnosis. 
                   - the number of male participants, and their mean age, median age, minimum and maximum age
                   - the number of female participants, and their mean age, median age, minimum and maximum age.
                   - 
                If any of the information is missing, return `null` for that field.               

                Text sample: ${text}
                """

output_schema = {
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
    }

ZERO_SHOT_MULTI_GROUP_FC = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": base_message + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": output_schema
}

ZERO_SHOT_MULTI_GROUP_FW_JSON = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": f"{base_message} \n Please, ensure to responsd in JSON format using the following JSON schema: {output_schema}"
        }
    ],
    "output_schema": output_schema,
    "response_format": {"type": "json_object", "schema": output_schema},
}

ZERO_SHOT_MULTI_GROUP_OAI_JSON = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": f"{base_message} \n Please, ensure to responsd in JSON format using the following JSON schema: {output_schema}"
        }
    ],
    "output_schema": output_schema,
    "response_format": {"type": "json_object"},
}


def clean_predictions(predictions):
    # Clean known issues with GPT demographics predictions
    predictions = [p for p in predictions if "groups" in p]

    meta_keys = ["pmcid", "rank", "start_char", "end_char"]
    meta_keys = [k for k in meta_keys if k in predictions[0]]
    
    # Convert JSON to DataFrame
    predictions = pd.json_normalize(
        predictions, record_path=["groups"],
        meta=meta_keys
        )

    predictions = predictions.fillna(value=np.nan)
    predictions["group_name"] = predictions["group_name"].fillna("healthy")

    # If group name is healthy, blank out diagnosis
    predictions.loc[predictions.group_name == "healthy", "diagnosis"] = np.nan
    predictions = predictions.replace(0.0, np.nan)

    # Drop rows where count is NA
    predictions = predictions[~pd.isna(predictions["count"])]

    # Set group_name to healthy if no diagnosis
    predictions.loc[
        (predictions["group_name"] != "healthy") & (pd.isna(predictions["diagnosis"])),
        "group_name",
    ] = "healthy"

    # If no male count, substract count from female count columns
    ix_male_miss = (pd.isna(predictions["male count"])) & ~(
        pd.isna(predictions["female count"])
    )
    predictions.loc[ix_male_miss, "male count"] = (
        predictions.loc[ix_male_miss, "count"]
        - predictions.loc[ix_male_miss, "female count"]
    )

    # Same for female count
    ix_female_miss = (pd.isna(predictions["female count"])) & ~(
        pd.isna(predictions["male count"])
    )
    predictions.loc[ix_female_miss, "female count"] = (
        predictions.loc[ix_female_miss, "count"]
        - predictions.loc[ix_female_miss, "male count"]
    )

    return predictions