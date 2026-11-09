"""Structured output, stage 2 (SPECULATION "Structured output, stage 2"): with RAGON_STRUCTURED=1 the chat route
honours response_format (json_object, json_schema) and tool_choice "required" by masking the sampler.

Every test skips unless the server has structured output on. The probe: a schema keyword outside the stage-1
subset ("pattern") is answered 400 unsupported_json_schema by a structured server and ignored (200) by one
without it."""
import json

import pytest

from conftest import chat_text, stream_chat_parts

NO_THINK = {"enable_thinking": False}
CITY = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"},
        "population": {"type": "integer"}, "coastal": {"type": "boolean"}, "districts": {
            "type": "array", "items": {"type": "string"}}},
        "required": ["city", "country", "population", "coastal", "districts"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}},
         {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                         "required": ["command"]}}}]


def schema_format(schema):
    return {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": schema}}


@pytest.fixture(scope="module")
def structured(server):
    status, out = server.request("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 1, "chat_template_kwargs": NO_THINK,
        "response_format": schema_format({"type": "string", "pattern": "^a$"})})
    if status != 400:
        pytest.skip("structured output is off on this server (RAGON_STRUCTURED=1 not set)")
    return server


def check_city(text):
    obj = json.loads(text)
    assert set(obj) == set(CITY["required"])
    assert isinstance(obj["city"], str) and isinstance(obj["country"], str)
    assert isinstance(obj["population"], int) and not isinstance(obj["population"], bool)
    assert isinstance(obj["coastal"], bool)
    assert isinstance(obj["districts"], list) and all(isinstance(d, str) for d in obj["districts"])


def test_unsupported_schema_is_refused(structured):
    status, out = structured.request("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "chat_template_kwargs": NO_THINK,
        "response_format": schema_format({"type": "string", "pattern": "^a$"})})
    assert status == 400
    assert out["error"]["code"] == "unsupported_json_schema"
    assert "pattern" in out["error"]["message"]


@pytest.mark.parametrize("prompt", ["Tell me about Lisbon.", "Write a haiku about rain."])
def test_json_schema_greedy(structured, prompt):
    """Even a prompt that asks for prose gets an object of the schema."""
    out = structured.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 400,
        "response_format": schema_format(CITY), "chat_template_kwargs": NO_THINK})
    assert out["choices"][0]["finish_reason"] == "stop", chat_text(out)
    check_city(chat_text(out))


def test_json_schema_sampled(structured):
    for seed in range(3):
        out = structured.post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Describe a city you like, with three of its districts."}],
            "temperature": 1.0, "top_p": 0.95, "seed": seed, "max_tokens": 1000, "response_format": schema_format(CITY),
            "chat_template_kwargs": NO_THINK})
        assert out["choices"][0]["finish_reason"] == "stop", chat_text(out)
        check_city(chat_text(out))


def test_json_schema_streamed(structured):
    # The prompt bounds the list: greedy decoding can repeat inside an unbounded array until max_tokens (seen
    # 2026-11-09: "Tell me about Osaka." listed prefectures in a loop); test_bounded_array_stops bounds it by maxItems.
    status, events = structured.stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Tell me about Osaka, with three of its districts."}],
        "temperature": 0, "max_tokens": 1000, "stream": True, "response_format": schema_format(CITY), "chat_template_kwargs": NO_THINK})
    assert status == 200
    content, _, _, finishes, _, done = stream_chat_parts(events)
    assert done and finishes[-1] == "stop", content
    check_city(content)


def test_bounded_array_stops(structured):
    """Stage 3: with maxItems the grammar closes the array after its last item, so the prompt that looped above
    without a bound in it ends, with at most three districts."""
    schema = json.loads(json.dumps(CITY))
    schema["properties"]["districts"]["maxItems"] = 3
    out = structured.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Tell me about Osaka."}], "temperature": 0, "max_tokens": 1000,
        "response_format": schema_format(schema), "chat_template_kwargs": NO_THINK})
    assert out["choices"][0]["finish_reason"] == "stop", chat_text(out)
    check_city(chat_text(out))
    assert len(json.loads(chat_text(out))["districts"]) <= 3


def test_json_schema_after_thinking(structured):
    """The matcher arms after </think>: the reasoning is free, the answer follows the schema."""
    out = structured.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Pick a European capital and describe it briefly."}],
        "temperature": 0, "max_tokens": 4000, "response_format": schema_format(CITY)})
    assert out["choices"][0]["finish_reason"] == "stop", chat_text(out)
    check_city(chat_text(out))


def test_json_object(structured):
    out = structured.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "List three colours."}], "temperature": 0, "max_tokens": 300,
        "response_format": {"type": "json_object"}, "chat_template_kwargs": NO_THINK})
    assert out["choices"][0]["finish_reason"] == "stop", chat_text(out)
    assert isinstance(json.loads(chat_text(out)), dict)


@pytest.mark.parametrize("prompt", ["Hello! How are you today?", "What is the weather in Oslo?"])
def test_tool_choice_required(structured, prompt):
    """A chit-chat prompt still yields a call; string values never hold a stray </parameter."""
    out = structured.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}], "tools": TOOLS, "tool_choice": "required",
        "temperature": 0, "max_tokens": 300, "chat_template_kwargs": NO_THINK})
    choice = out["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    assert calls and choice["finish_reason"] == "tool_calls"
    for c in calls:
        assert c["function"]["name"] in {t["function"]["name"] for t in TOOLS}
        args = json.loads(c["function"]["arguments"])
        assert isinstance(args, dict) and "</parameter" not in c["function"]["arguments"]


def test_tool_choice_auto_unchanged(structured):
    """Without "required" the request is not constrained: a greeting is answered in text."""
    out = structured.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Hello! How are you today?"}], "tools": TOOLS,
        "temperature": 0, "max_tokens": 200, "chat_template_kwargs": NO_THINK})
    assert chat_text(out).strip()


def test_logprobs_stay_raw(structured):
    """The mask steers the sampler only: the reported logprobs are the model's. A prose prompt's first token under
    a schema has the same top_logprobs as without one, and the token the grammar forces carries its raw logprob."""
    def first(fmt):
        body = {"messages": [{"role": "user", "content": "Write a haiku about rain."}], "temperature": 0,
                "max_tokens": 3, "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": NO_THINK}
        if fmt:
            body["response_format"] = fmt
        return structured.post("/v1/chat/completions", body)["choices"][0]["logprobs"]["content"][0]
    # top_logprobs lists the chosen token first, then the most likely others (api.cpp top_logprobs).
    free, held, again = first(None), first(schema_format(CITY)), first(None)
    top = lambda e: [(t["token"], t["logprob"]) for t in e["top_logprobs"]]
    assert top(free) == top(again)  # the comparison below is only as good as this
    raw = dict(top(free))
    assert held["token"] != free["token"], (free["token"], held["token"])  # the grammar overrode the argmax
    others = [(t, lp) for t, lp in top(held) if t != held["token"]]
    print(f"\nlogprobs_stay_raw: free {free['token']!r} {free['logprob']:.6f}, held {held['token']!r} "
          f"{held['logprob']:.6f}; held others {others}")
    assert others and others == [(t, raw[t]) for t, _ in others], (top(free), top(held))  # bit-identical
    assert others == top(free)[:len(others)]
    assert held["logprob"] == top(held)[0][1] and held["logprob"] < top(free)[-1][1]  # raw, not renormalised
