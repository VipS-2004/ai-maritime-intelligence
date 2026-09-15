import json
import os
import time

from google import genai
from google.genai import types


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini rejects a request due to quota exhaustion."""


def _is_quota_error(error):
    """
    Detect quota / rate-limit failures from a Gemini API error.

    Checks for HTTP 429 status codes and the RESOURCE_EXHAUSTED
    status used by the Gemini API for quota exhaustion. String
    matching is kept as a fallback for wrapped exceptions.
    """

    if getattr(error, "code", None) == 429:
        return True

    status = getattr(error, "status", None)

    if isinstance(status, str) and status.upper() == "RESOURCE_EXHAUSTED":
        return True

    message = str(error).lower()

    return (
        "resource_exhausted" in message
        or "quota" in message
    )


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

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30000
        ),
    )

    analysis_data = json.dumps(
        analysis,
        indent=2,
        default=str
    )

    prompt = f"""
Analyze the following maritime computer vision observations:

{analysis_data}

Return the assessment in Markdown using exactly these four sections:

### Intelligence Assessment

Provide a concise summary of the detected maritime situation.
Mention the main vessel composition, traffic conditions, risk level,
and spatial observations supported by the data.
When the data includes a "spatial" block, incorporate the zone counts,
primary hotspot, and maximum zone concentration into the assessment.

### Risk Explanation

Explain why the computer vision analysis produced the given risk level.
Use only the provided evidence.

### Areas Requiring Attention

List the observations that may deserve human analyst verification.
Do not present assumptions as facts.

### Recommended Action

Provide practical analyst verification steps based only on the available
evidence. Do not recommend actions based on unsupported assumptions.

Keep each section concise. Do not add additional sections.
"""

    last_error = None

    for attempt in range(3):

        try:

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

        except Exception as error:

            if _is_quota_error(error):

                raise GeminiQuotaError(
                    "Gemini quota exhausted for this API key. "
                    f"Details: {error}"
                ) from error

            last_error = error

            if attempt < 2:

                delay = 3 * (2 ** attempt)

                time.sleep(delay)

    raise RuntimeError(
        f"Gemini intelligence generation failed after 3 attempts: "
        f"{last_error}"
    )