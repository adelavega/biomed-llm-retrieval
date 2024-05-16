import pandas as pd
import numpy as np


def clean_predictions(predictions):
    # Clean known issues with GPT demographics predictions
    predictions = [p for p in predictions if "groups" in p]

    meta_keys = ["pmcid", "rank", "start_char", "end_char", "id"]
    meta_keys = [k for k in meta_keys if k in predictions[0]]
    
    # Convert JSON to DataFrame
    predictions = pd.json_normalize(
        predictions, record_path=["groups"],
        meta=meta_keys
        )
    
    predictions.columns = predictions.columns.str.replace(' ', '_')

    predictions = predictions.fillna(value=np.nan)
    predictions["group_name"] = predictions["group_name"].fillna("healthy")

    # Drop rows where count is NA
    predictions = predictions[~pd.isna(predictions["count"])]

    # Set group_name to healthy if no diagnosis
    predictions.loc[
        (predictions["group_name"] != "healthy") & (pd.isna(predictions["diagnosis"])),
        "group_name",
    ] = "healthy"

    # If no male count, substract count from female count columns
    ix_male_miss = (pd.isna(predictions["male_count"])) & ~(
        pd.isna(predictions["female_count"])
    )
    predictions.loc[ix_male_miss, "male_count"] = (
        predictions.loc[ix_male_miss, "count"]
        - predictions.loc[ix_male_miss, "female_count"]
    )

    # Same for female count
    ix_female_miss = (pd.isna(predictions["female_count"])) & ~(
        pd.isna(predictions["male_count"])
    )
    predictions.loc[ix_female_miss, "female_count"] = (
        predictions.loc[ix_female_miss, "count"]
        - predictions.loc[ix_female_miss, "male_count"]
    )

    return predictions