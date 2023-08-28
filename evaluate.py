""" Helper functions to evaluate the performance of the entity extraction models """

# How accurately does it predict the # of groups for each study?
from collections import defaultdict
import pprint
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score

def isin(a, li):
    """ Nan safe is in function,  that returns second value.
    This is useful because np.nan != np.nan 
    """
    for b in li:
        if (pd.isna(a) & pd.isna(b)) or a == b:
            return b

    return False

def score_columns(df1, df2,  scoring='mpe'):
    """ Score columns of two dataframes, aggregatingby pmcid. """

    if scoring == 'mpe':
        scorer = mean_absolute_percentage_error
    elif scoring == 'r2':
        scorer = r2_score

    res_mean = {}
    res_sum = {}
    counts = {}
    # Only look at columns if dtype is int or float
    df1 = df1.select_dtypes(include=[np.int64, np.float64])
    df2 = df2.select_dtypes(include=[np.int64, np.float64])

    df1_sum = df1.groupby('pmcid').sum()
    df2_sum = df2.groupby('pmcid').sum()

    df1_mean = df1.groupby('pmcid').mean()
    df2_mean = df2.groupby('pmcid').mean()

    for col in df1_mean:
        # Compute overlap using mean
        # Using mean results in NAs when aggregated values are both nan
        df1_mean_col = df1_mean[col].dropna()
        df2_mean_col = df2_mean[col].dropna()
        
        # Index of rows where both df1 and df2 are not nan
        ix = df1_mean_col.index.intersection(df2_mean_col.index)

        overlap = np.round(len(ix) / df2_mean_col.shape[0], 2)

        # Mean aggregation
        df1_mean_col = df1_mean_col.loc[ix]
        df2_mean_col = df2_mean_col.loc[ix]

        # Sum aggregation
        df1_sum_col = df1_sum[col].loc[ix]
        df2_sum_col = df2_sum[col].loc[ix]

        res_mean[col] = np.round(
            scorer(df1_mean_col, df2_mean_col),
            2
        )

        res_sum[col] = np.round(
            scorer(df1_sum_col, df2_sum_col),
            2
        )

        counts[col] = overlap

    return res_mean, res_sum, counts


def compare_by_pmcid(df1, df2):
    res = defaultdict(float)
    for pmcid, df in df1.groupby('pmcid'):
        for col in df:
            if col != 'pmcid':
                match_df2 = df2[df2.pmcid == pmcid][col].to_list()
                score = 0
                for v in df[col]:
                    rem_val = isin(v, match_df2)
                    if rem_val:
                        score += 1
                        match_df2.remove(rem_val)
                res[col] += score

    res = {k: np.round(1 - (v / len(df1)), 2) for k,v in res.items()}
    
    return res
    
def evaluate_predictions(predictions, annotations):
    # Subset to only pmcids in predictions
    diff = set(set(annotations.pmcid.unique()) - set(predictions.pmcid.unique()))
    annotations = annotations[annotations.pmcid.isin(predictions.pmcid.unique())]

    pred_n_groups = predictions.groupby('pmcid').size()
    n_groups = annotations.groupby('pmcid').size()
    correct_n_groups = (n_groups == pred_n_groups)
    more_groups_pred = (n_groups < pred_n_groups)
    less_groups_pred = (n_groups > pred_n_groups)
    
    ix_corr_n_groups = correct_n_groups[correct_n_groups == True].index
    ix_more_groups =  more_groups_pred[more_groups_pred == True].index
    ix_less_groups = less_groups_pred[less_groups_pred == True].index

    print(f"Exact match # of groups: {correct_n_groups.mean():.2f}\n",
    f"More groups predicted: {more_groups_pred.mean():.2f}\n",
    f"Less groups predicted: {less_groups_pred.mean():.2f}\n",
    f"Missing pmcids: {diff}\n"
    )

    # Compare by columns (matched accuracy)
    print("Column wise comparison of predictions and annotations (error):\n")
    pprint.pprint(compare_by_pmcid(annotations, predictions))


    res_mean, res_sum, counts = score_columns(annotations, predictions, scoring='mpe')

    # Compare by columns (count of pmcids with overlap)
    print("\nPercentage response given by pmcid:\n")
    pprint.pprint(counts)
    

    # Compare by columns (summed mean percentage error on sum by pmcid)
    print("\nSummed Mean percentage error:\n")
    pprint.pprint(res_mean)

    # Compare by columns (summed mean percentage error on mean by pmcid)
    print("\nAveraged Mean percentage error:\n")
    pprint.pprint(res_sum)

    return ix_corr_n_groups, ix_more_groups, ix_less_groups

def clean_predictions(predictions):
    # Clean known issues with GPT predictions

    predictions = predictions.copy()
    
    predictions = predictions.fillna(value=np.nan)
    predictions['group_name'] = predictions['group_name'].fillna('healthy')

    # If group name is healthy, blank out diagnosis
    predictions.loc[predictions.group_name == 'healthy', 'diagnosis'] = np.nan
    predictions = predictions.replace(0.0, np.nan)

    # Drop rows where count is NA
    predictions = predictions[~pd.isna(predictions['count'])]

    # Set group_name to healthy if no diagnosis
    predictions.loc[(predictions['group_name'] != 'healthy') & (pd.isna(predictions['diagnosis'])), 'group_name'] = 'healthy'

    # If no male count, substract count from female count columns
    ix_male_miss = (pd.isna(predictions['male count'])) & ~(pd.isna(predictions['female count']))
    predictions.loc[ix_male_miss, 'male count'] = predictions.loc[ix_male_miss, 'count'] - predictions.loc[ix_male_miss, 'female count']

    # Same for female count
    ix_female_miss = (pd.isna(predictions['female count'])) & ~(pd.isna(predictions['male count']))
    predictions.loc[ix_female_miss, 'female count'] = predictions.loc[ix_female_miss, 'count'] - predictions.loc[ix_female_miss, 'male count']

    return predictions