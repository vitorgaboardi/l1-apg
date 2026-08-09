TOOLALPACA_PERSONA_PROMPT_UTTERANCE_GENERATION = """
You are the following persona: {persona_description}

Imagine that you want to use the features provided by APIs in your daily life. Your task is to come up with realistic scenarios for using these APIs and express them as natural language instructions, as if you were asking a friend or assistant for help.

Please follow these guidelines for generating the instructions:
1. Use a mix of interrogative sentences, first-person statements, imperative sentences, and other structures that convey a request. Aim for diversity in your instructions.
2. Do not mention the API's name in your instructions.
3. Your instructions should only involve the features provided by these APIs. 
4. Use specific nouns and real-world examples from various domains, such as entertainment, sports, or technology. Avoid using any form of placeholder or generic phrases, such as "this xxx", "a xxx" or "a specific xxx", and provide concrete details instead.
5. Try not to repeat the verb for each instruction to maximize diversity.
6. Ensure diversity in language by combining questions with imperative statements and other structures that convey a request.
7. The instructions must be written in English.

Please follow these guidelines for generating API Calls:
- Each instruction must have a corresponding API call extracting all values stated.
- Values must follow the API specification exactly (ISO dates, standardized codes, enumerated values) even when the instruction uses natural language.
- Every semantic value present in the instruction must be mapped in the API call arguments.
- Do not invent argument values.

<API>
{api_method_description}
</API>

Based on the API provided above, generate {number_of_utterances} natural language instructions and corresponding API call following the guidelines.
Remember, You MUST use the persona by considering the persona's background, preferences, and context when generating the instructions. The instructions should reflect the persona's unique perspective and needs.

OUTPUT FORMAT:
Return only a JSON object with no additional text:
{{
    "data": [
        {{"utterance": "...", "api_call": [{{"name": "method_name", "arguments": {{"param": "value"}}}}]}},
        ...
    ]
}}
"""