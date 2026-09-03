import json
import os

from google import genai
from google.genai import types


SYSTEM_PROMPT = """
You are a Maritime Intelligence Analyst assisting a human analyst.

You receive structured observations produced by a computer vision
system analyzing satellite imagery.

Your task is to explain the observations clearly and conservatively.

Follow these rules:

1. Report detected facts accurately.
2. Explain the risk level using only the provided analysis.
3. Identify observations that may require analyst attention.
4. Clearly distinguish computer vision observations from analyst interpretation.
5. Treat vessel proximity or clustering only as a spatial observation.
6. Never claim that clustering proves coordination, formation, intent,
   hostile activity, military operations, or suspicious behavior.
7. Never invent vessel identities, locations, coordinates, events,
   missions, or external intelligence.
8. Do not assume that military and civilian vessels are interacting
   simply because they appear close together.
9. When evidence is insufficient, explicitly say that further
   verification is required.
10. Recommended actions should be framed as analyst verification steps,
    not as conclusions.

Keep the assessment concise, professional, and evidence-based.
"""


def generate_intelligence_report(analysis):
    """
    Generate an AI-powered maritime intelligence assessment
    from structured computer vision analysis.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not configured."
        )

    client = genai.Client(api_key=api_key)

    analysis_data = json.dumps(
        analysis,
        indent=2,
        default=str
    )

    prompt = f"""
Analyze the following maritime computer vision observations:

{analysis_data}

Produce an intelligence assessment with these sections:

INTELLIGENCE ASSESSMENT
RISK EXPLANATION
AREAS REQUIRING ATTENTION
RECOMMENDED ACTION
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            SYSTEM_PROMPT,
            prompt
        ],
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    )

    return response.text
