from labelrepo.projects.participant_demographics import \
        get_participant_demographics

import pandas as pd

subgroups = get_participant_demographics()

top_raters = ['Jerome_Dockes', 'kailu_song', 'calvin_surbey', 'joon_hong', 'ju-chi_yu']

combined = []
for pmcid, group in subgroups.groupby('pmcid'):
    chosen = False
    for rater in top_raters:
        if rater in group.annotator_name.values:
            combined.append(group[group.annotator_name == rater])
            chosen = True
            break

    if not chosen:
        # If not choose rating with most counts
        chosen_rater = group.groupby('annotator_name')['count'].sum().idxmax()
        combined.append(group[group.annotator_name == chosen_rater])

combined = pd.concat(combined)

combined.to_csv('annotations/combined_pd.csv')
