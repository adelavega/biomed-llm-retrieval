from .schemas import BaseDemographicsSchema, RelevantSentSchema, FewShot2Schema

base_message = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

Your task is to identify groups of participants that participated in the study, and underwent MRI.
If there is no mention of any participant groups, return a null array.

For each group identify:
    - the number of participants in each group, and the diagnosis.
    - the number of male participants, and their mean age, median age, minimum and maximum age
    - the number of female participants, and their mean age, median age, minimum and maximum age.

    If any of the information is missing, return `null` for that field.         

Text sample: ${text}
"""

ZERO_SHOT_MULTI_GROUP_FC = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": base_message + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": BaseDemographicsSchema.model_json_schema()
}


base_message_ft = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

Your task is to identify groups of participants that participated in the study, and underwent MRI.
If there is no mention of any participant groups, return a null array.

For each group identify:
    - the number of participants in each group, and the diagnosis.
    - the number of male participants, and their mean age, median age, minimum and maximum age
    - the number of female participants, and their mean age, median age, minimum and maximum age.

Be as accurate as possible, and report information (especially diagnosis) using the technical terms (and abbreviations) used in the article.
If any of the information is missing, return `null` for that field.

Text sample: ${text}
"""

ZERO_SHOT_MULTI_GROUP_FTSTRICT_FC = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": base_message_ft + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": BaseDemographicsSchema.model_json_schema()
}

base_message_2 = """
You are an expert scientist in the field of neuroimaging.
You will be provided with a text sample from a scientific journal.
Your job is to accurately annotate the demographics of participants in scientific studies.
You will be paid a 500 dollar tip for achieving a 90 percent accuracy rate.

Your task is to identify groups of human volunteers that participated in the study, and underwent MRI.
If there is no mention of any participant groups, return a null array.

For each group identify:
    - the number of participants in each group, and the diagnosis.
    - the number of male participants, and their mean age, median age, minimum and maximum age
    - the number of female participants, and their mean age, median age, minimum and maximum age.

If *any* of the information is missing and you are not confident, return null for that specific field.
ONLY report subgroups that underwent MRI, fMRI or neuroimaging.

Text sample: ${text}
"""

ZERO_SHOT_MULTI_GROUP_FC_2 = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": base_message_2 + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": RelevantSentSchema.model_json_schema()
}




few_shot_message = """
You will be provided with a text sample from a scientific journal.

Identify groups of participants in the study.
If there is no mention of any participant groups in the text sample, return a null array, but do so only if there is no relevant information at all.

For each group identify:
    - the number of participants in each group, and the diagnosis.
    - the number of male participants, and their mean age, median age, minimum and maximum age
    - the number of female participants, and their mean age, median age, minimum and maximum age.
    - if this group of participants underwent MRI, fMRI or neuroimaging procedure.

If any participants are excluded from the final sample, report the final sample size.
If any of the information is missing, return `null` for that field.

Here is an example of a correct response.

Example text sample:

Subjects. 
Behavioral Study

For the behavioral task, twenty healthy participants (7 men, 13 women; mean age 25.5±3 years) volunteered to participate. Handedness was determined using a test for manual dominance [57]. 
None reported visual or psychomotor dysfunction. All subjects were naïve as to the experimental design or purpose and gave their informed consent to participate in the study. 
The experimental procedures were approved by the Institutional Review Board at the University of Padua and were in accordance with the declaration of Helsinki.

Functional MRI
Nineteen healthy controls (12 women and 7 men; age range: 19–30 years; mean age: 24,7 years) and fifteen schizophrenic patients (10 women and 5 men; age range: 21–35 years; mean age: 26,1 years) participated in the experiment. 
Three healthy participants were excluded from the final sample due to excessive motion. They all had normal or corrected-to-normal vision. 
All patients were medicated and had a history of prior hospitalizations due to their symptoms. 
Control subjects were recruited from advertisements and group matched to patients on age, gender, socioeconomic status (SES), parental socio-economic status (PSES), and handedness (Table 1). 
Inclusion criteria for both patients and controls were as follows: right handedness, ages between 20 and 60 years, no hearing impairments, no history of electroconvulsive therapy, no history of neurological illness, no substance abuse or dependence history during the last 5 years (assessed using the addiction severity index and the Diagnostic and Statistical Manual of Mental Disorders (DSM)-IV criteria), no alcohol use 24 h prior to testing, verbal IQ above 75, and English as a first language.

Response:

{
    "groups": [
        {
            "group_name": "healthy",
            "subgroup_name": "behavioral",
            "diagnosis": null,
            "count": 20,
            "male count": 7.0,
            "female count": 13.0,
            "age mean": 25.53,
            "age minimum": null,
            "age maximum": null,
            "age median": null,
            "imaging_sample": false
        },
        {
            "group_name": "healthy",
            "subgroup_name": null,
            "diagnosis": null,
            "count": 16,
            "male count": 7.0,
            "age mean": 24.7,
            "female count": 12.0,
            "age minimum": 19.0,
            "age maximum": 30.0,
            "age median": null,
            "imaging_sample": true
        },
        {
            "group_name": "patients",
            "subgroup_name": null,
            "diagnosis": "schizophrenia",
            "count": 15,
            "male count": 5.0,
            "age mean": 26.1,
            "female count": 10.0,
            "age minimum": 21.0,
            "age maximum": 35.0,
            "age median": null,
            "imaging_sample": true
        }
    ]
}


Now it's your turn.

Text sample: ${text}
"""

FEW_SHOT_FC = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": few_shot_message + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": BaseDemographicsSchema.model_json_schema()
}


few_shot_message_2 = """
You will be provided with a text sample from a scientific journal.

Identify groups of participants in the study.
If there is no mention of any participant groups in the text sample, return a null array, but do so only if there is no relevant information at all.

For each group identify:
    - the final number of participants in each group, and the diagnosis.
    - the number of male participants, and their mean age, median age, minimum and maximum age
    - the number of female participants, and their mean age, median age, minimum and maximum age.
    - if this group of participants only underwent a behavioral task, or if they underwent MRI, fMRI or neuroimaging procedure.

If any participants are excluded from the final sample, report only the final sample size.
If any of the information is missing, return `null` for that field.

Here is an example of a correct response.

Text sample:

Subjects.

Twenty healthy participants participated in the behavioral assesment. Handedness was determined using a test for manual dominance [57]. 
None reported visual or psychomotor dysfunction. Subjects were separated into left handed (3 men, 5 women; mean age 25.5±3 years) and right handed groups (4 men, 8 women; mean age 25.5±3 years)). All subjects were naïve as to the experimental design or purpose and gave their informed consent to participate in the study. 
The experimental procedures were approved by the Institutional Review Board at the University of Padua and were in accordance with the declaration of Helsinki.

Nineteen healthy controls (12 women and 7 men; age range: 19–30 years; mean age: 24,7 years) and fifteen schizophrenic patients (10 women and 5 men; age range: 21–35 years; mean age: 26,1 years) participated in the experiment. 
Three healthy participants were excluded from the final sample due to excessive motion. They all had normal or corrected-to-normal vision. 
All patients were medicated and had a history of prior hospitalizations due to their symptoms. 
Control subjects were recruited from advertisements and group matched to patients on age, gender, socioeconomic status (SES), parental socio-economic status (PSES), and handedness (Table 1). 
Inclusion criteria for both patients and controls were as follows: right handedness, ages between 20 and 60 years, no hearing impairments, no history of electroconvulsive therapy, no history of neurological illness, no substance abuse or dependence history during the last 5 years (assessed using the addiction severity index and the Diagnostic and Statistical Manual of Mental Disorders (DSM)-IV criteria), no alcohol use 24 h prior to testing, verbal IQ above 75, and English as a first language.

Response:

{
    "groups": [
        {
            "group_name": "healthy",
            "subgroup_name": "left handed",
            "diagnosis": null,
            "count": 20,
            "male count": 7.0,
            "female count": 13.0,
            "age mean": 25.53,
            "age minimum": null,
            "age maximum": null,
            "age median": null,
            "assesment_type": "behavioral"
        },
        {
            "group_name": "healthy",
            "subgroup_name": "right handed",
            "diagnosis": null,
            "count": 20,
            "male count": 7.0,
            "female count": 13.0,
            "age mean": 25.53,
            "age minimum": null,
            "age maximum": null,
            "age median": null,
            "assesment_type": "behavioral"
        },
        {
            "group_name": "healthy",
            "subgroup_name": null,
            "diagnosis": null,
            "count": 16,
            "male count": 7.0,
            "age mean": 24.7,
            "female count": 12.0,
            "age minimum": 19.0,
            "age maximum": 30.0,
            "age median": null,
            "imaging_sample": "imaging"
        },
        {
            "group_name": "patients",
            "subgroup_name": null,
            "diagnosis": "schizophrenia",
            "count": 15,
            "male count": 5.0,
            "age mean": 26.1,
            "female count": 10.0,
            "age minimum": 21.0,
            "age maximum": 35.0,
            "age median": null,
            "imaging_sample": "imaging"
        }
    ]
}


Now it's your turn.

Text sample: ${text}
"""


FEW_SHOT_FC_2 = {
    "search_query": "How many participants or subjects were recruited for this study?",
    "messages": [
        {
            "role": "user",
            "content": few_shot_message_2 + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": FewShot2Schema.model_json_schema()
}



