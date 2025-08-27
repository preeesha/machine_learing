import re

def clean_json_string(s: str) -> str:
    """Clean JSON string from LLM response."""
    s = re.sub(r"^```(?:json)?\n", "", s)
    s = re.sub(r"\n```$", "", s)
    return s.strip()
