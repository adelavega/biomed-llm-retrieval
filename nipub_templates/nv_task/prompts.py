from .schemas import StudyMetadataModel

base_message = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

Your task is to identify information about the design of the fMRI task and analysis of the neuroimaging data.
If any information is missing or not explicitly stated in the text, return `null` for that field.

For any extracted text, maintain fidelity to the source. Avoid inferring information not explicitly stated. If a field cannot be completed, return `null`.

Text sample: ${text}
"""


ZERO_SHOT_TASK = {
    "messages": [
        {
            "role": "user",
            "content": base_message + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": StudyMetadataModel.model_json_schema()
}