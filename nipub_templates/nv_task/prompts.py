from .schemas import TaskMetadataModel

base_message = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

Your task is to identify information about the design of the fMRI task and analysis of the neuroimaging data.
For any information that is missing, return `null`.

Text sample: ${text}
"""


ZERO_SHOT_TASK = {
    "messages": [
        {
            "role": "user",
            "content": base_message + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": TaskMetadataModel.model_json_schema()
}