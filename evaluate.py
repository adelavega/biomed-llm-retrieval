""" Helper functions to evaluate the performance of the entity extraction models """

# How accurately does it predict the # of groups for each study?
from collections import defaultdict
import pprint
import numpy as np
import pandas as pd

def isin(a, li):
    """ Nan safe is in function,  that returns second value.
    This is useful because np.nan != np.nan 
    """
    for b in li:
        if (pd.isna(a) & pd.isna(b)) or a == b:
            return b

    return False

def compare_by_pmcid(df1, df2):
    """ Compute # of matches of values for each pmcid, for each column """
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

    res = {k: np.round(v / len(df1), 2) for k,v in res.items()}
    
    return res
    
def evaluate_predictions(predictions, annotations):
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
    f"Less groups predicted: {less_groups_pred.mean():.2f}\n"
    )

    print("Column wise comparison of predictions and annotations:\n")

    # Compare by columns
    pprint.pprint(compare_by_pmcid(annotations, predictions))
    
    return ix_corr_n_groups, ix_more_groups, ix_less_groups