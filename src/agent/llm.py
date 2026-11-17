from dotenv import load_dotenv
import os

from google import genai
from google.genai import types

load_dotenv()

try:
    _http = types.HttpOptions(timeout=40_000,
                              retry_options=types.HttpRetryOptions(attempts=1))
    client = genai.Client(http_options=_http)
except Exception:
    client = genai.Client()


def _chain(single_var, list_var, default):
    first = os.getenv(single_var)
    models = [m.strip() for m in os.getenv(list_var, default).split(",") if m.strip()]
    return ([first] if first else []) + [m for m in models if m != first]


# Cheap, fast models plan the request; stronger ones write the answer.
# If one model's free quota is used up, the next one in the list is tried.
ROUTER_MODELS = _chain("ROUTER_MODEL", "ROUTER_MODELS",
                       "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash")
ANSWER_MODELS = _chain("ANSWER_MODEL", "ANSWER_MODELS",
                       "gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash,gemini-3.5-flash-lite")

_CACHE = {} 


def generate(models, contents, config):
    key = (tuple(models), contents, str(config.system_instruction))
    if key in _CACHE:
        return _CACHE[key]
    last = None
    for m in models:
        try:
            resp = client.models.generate_content(model=m, contents=contents, config=config)
            if len(_CACHE) > 500:
                _CACHE.clear()
            _CACHE[key] = resp
            return resp
        except Exception as e:
            last = e
            print(f"LLM error on {m}: {str(e)[:160]}")
    raise last
