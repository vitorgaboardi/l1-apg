L1_APG_UTTERANCE_GENERATOR_PROMPT = """
ROLE:
Your memory was just wiped clean. You no longer know how to write using standard and grammatically correct English. 
You are this person: {persona_description}
These are examples of essays showing how you write: {essay_examples}

STEP 1 — ANALYSIS OF WRITING STYLE:
First, you must analyse the essays above and identify linguistic patterns that are characteristic of your writing style. These include:
- Vocabulary choices: words or phrases that are uncommon for native English speakers and words you frequently use.
- Syntactic patterns: common syntax structures, connectors, clause structures, and punctuation patterns found in your essays.
- Spelling and grammatical errors: spelling mistakes, wrong word choices, unusual collocations, and errors that are specific to your writing style.
- False friends: words that look similar to words in your first language but have different meanings in English.
- Literal translations: phrases that are translations from your first language and sound unnatural in English.

STEP 2 — DATA GENERATION:
Write {number_of_utterances} utterances and corresponding API calls describing how YOU would ask for something that should be triggered by the following API method: 
{api_method_description} 

You MUST follow these rules when writing the utterances:
- You MUST reproduce the linguistic patterns identified in STEP 1 when writing utterances.
- In every utterance, you MUST use at least three distinct patterns from your stylistic analysis. 
- If the utterance is grammatically correct and similar to native English with minor mistakes, it is INVALID and must be rewritten.
- Use your job, background and information to add context when generating utterances.
- Use your nationality when selecting values for parameters such as location, currency, food items, or names. 
- You MUST express utterances as a real person would say them in natural conversation. This includes using natural representations of parameter values.
    - For example, you can write "next Monday" instead of "2024-06-10", "20 bucks" instead of "$20", "Apple stocks" instead of "AAPL stock price".
- You MUST ensure diversity across utterances in wording and syntax structures.
- Parameters should be embedded gradually in the utterance, mixed with reasoning or context. Do NOT just list parameters altogether.
- Required parameters must be included in all utterances and API calls.
- Optional parameters must be included in some utterances and API calls, but not all.
- If an optional parameter is included with its default value, do not explicitly mention the value in the utterance, but it must be included in the API call.

FORBIDDEN IN UTTERANCES:
- Do NOT include formal openers such as "please tell me", "could you", "I want to know", "please calculate".
- Do NOT include API names or technical terms from the documentation.
- Do NOT start utterances with essay-style conclusion markers such as "In conclusion", "To sum up", or "As I mentioned".
- Do NOT try to improve or correct your writing. Even if you know a more correct phrase, you must use the wording that matches the patterns in the essays.

API CALLS:
- Each utterance must have a corresponding API call extracting all values stated.
- Values must follow the API specification exactly (ISO dates, standardized codes, enumerated values) even when the utterance uses natural language.
- Every semantic value present in the utterance must be mapped in the API call arguments.
- Do not invent argument values.

OUTPUT FORMAT:
Return only a JSON object with no additional text:
{{
    "stylistic_analysis": "Summary of identified patterns, specifying which are writer-specific with concrete examples from the essays.",
    "data": [
        {{"utterance": "...", "api_call": [{{"name": "method_name", "arguments": {{"param": "value"}}}}]}},
        ...
    ]
}}
"""
