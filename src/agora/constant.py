SURVEY_PROMPT = """
    Respond to the following survey in valid JSON only, strictly following the schema and rules below.
    Rules

    Provide exactly one response per question.
    Use only the allowed Likert values listed below (exact spelling and casing).
    Do not output anything outside the JSON object.
    Make sure you answer to all of the questions.

    Allowed Likert values
    [
        "Strongly disagree",
        "Disagree",
        "Neutral",
        "Agree",
        "Strongly agree"
    ]

    JSON response schema
    {
        "Q1": "<Likert choice>",
        "Q2": "<Likert choice>",
        ...
    }
    Here are the questions:
    
"""

LIKERT_VALUES = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]

LIKERT_TO_SCORE = {
    "Strongly disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly agree": 2,
}