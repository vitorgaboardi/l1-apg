"""Tool-calling agent for downstream API-call evaluation (RQ3)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR    = PROJECT_ROOT / "data" / "apis"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "agent" / "apis"))

from converter_currency      import converter_currency
from get_current_weather     import get_current_weather
from get_exercises           import get_exercises
from get_geolocation         import get_geolocation
from get_movies              import get_movies
from get_nutrition           import get_nutrition
from get_stock_data          import get_stock_data
from get_weather_forecast    import get_weather_forecast
from search_paper            import search_paper
from search_restaurants      import search_restaurants
from search_touristic_places import search_touristic_places

from .utils import get_tools
from .prompt import SYSTEM_PROMPT_TEMPLATE


@dataclass
class AgentStep:
    thought:     str
    action_name: str
    action_args: dict
    observation: dict


@dataclass
class AgentResult:
    utterance:         str
    predicted_calls:   list[dict]        = field(default_factory=list)
    execution_results: list[dict]        = field(default_factory=list)
    steps:             list[AgentStep]   = field(default_factory=list)
    raw_response:      str               = ""
    final_text:        str               = ""
    error:             str | None        = None


class Agent:
    """Class-oriented agent that uses model-native tool calling."""

    # Registry: api function name → callable
    TOOL_REGISTRY: dict[str, Any] = {
        "converter_currency":      converter_currency,
        "get_current_weather":     get_current_weather,
        "get_exercises":           get_exercises,
        "get_geolocation":         get_geolocation,
        "get_movies":              get_movies,
        "get_nutrition":           get_nutrition,
        "get_stock_data":          get_stock_data,
        "get_weather_forecast":    get_weather_forecast,
        "search_paper":            search_paper,
        "search_restaurants":      search_restaurants,
        "search_touristic_places": search_touristic_places,
    }

    def __init__(
        self,
        client:      OpenAI,
        model:       str,
        tools_dir:   Path  = TOOLS_DIR,
        max_steps:   int   = 5,
        temperature: float = 0.0,
    ) -> None:
        self.client      = client
        self.model       = model
        self.max_steps   = max_steps
        self.temperature = temperature
        self._system_prompt = SYSTEM_PROMPT_TEMPLATE
        self._tools = get_tools(str(tools_dir))


    def run(self, utterance: str) -> AgentResult:
        """Run a tool-calling loop for a single utterance."""
        result = AgentResult(utterance=utterance)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": utterance},
        ]

        for _ in range(self.max_steps):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
            except Exception as exc:
                result.error = f"LLM call failed: {exc}"
                return result

            assistant_message = response.choices[0].message
            assistant_text = assistant_message.content or ""
            result.raw_response += assistant_text + "\n"

            tool_calls = assistant_message.tool_calls or []
            if not tool_calls:
                result.final_text = assistant_text
                return result

            tool_call_payloads = []
            for call in tool_calls:
                tool_call_payloads.append({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                })

            messages.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_call_payloads,
            })

            for call in tool_calls:
                api_name = call.function.name
                api_args_raw = call.function.arguments or "{}"
                try:
                    api_args = json.loads(api_args_raw)
                except json.JSONDecodeError:
                    api_args = {}

                observation = self._execute_tool(api_name, api_args)
                result.execution_results.append(observation)
                result.predicted_calls.append({"name": api_name, "arguments": api_args})
                result.steps.append(AgentStep(
                    thought=assistant_text.strip(),
                    action_name=api_name,
                    action_args=api_args,
                    observation=observation,
                ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": api_name,
                    "content": json.dumps(observation, ensure_ascii=False),
                })

        return result

    def _execute_tool(self, name: str, arguments: dict) -> dict:
        """Dispatch to the actual Python callable; catch all errors."""
        fn = self.TOOL_REGISTRY.get(name)
        if fn is None:
            return {"status_code": 404, "error": f"Unknown tool: '{name}'"}
        try:
            return fn(**arguments)
        except TypeError as exc:
            return {"status_code": 400, "error": f"Invalid arguments: {exc}"}
        except Exception as exc:
            return {"status_code": 500, "error": str(exc)}
