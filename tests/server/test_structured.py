"""Structured output, stages 2 and 4 (SPECULATION "Structured output, stage 2" and "stage 4"): unless
RAGON_STRUCTURED=0, every route masks the sampler: response_format on chat and completions, text.format on
/v1/responses, tool_choice "required" and named, parallel_tool_calls false, and a custom tool's grammar format.

Every test skips unless the server has structured output on. The probe: a schema keyword outside the stage-1
subset ("pattern") is answered 400 unsupported_json_schema by a structured server and ignored (200) by one
without it."""
import json
import re

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
        pytest.skip("structured output is off on this server (RAGON_STRUCTURED=0, or a build without RAGON_GRAMMAR)")
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


# Stage 4 (API §3/§4): the completions and Responses routes, the named and single-call forms, custom-tool grammars.
ANSWER = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
          "additionalProperties": False}
RESP_TOOLS = [{"type": "function", "name": t["function"]["name"], "description": t["function"]["description"],
               "parameters": t["function"]["parameters"]} for t in TOOLS]
NO_REASONING = {"effort": "none"}
SUPPORTED = ("is not supported (supported: type, properties, required, additionalProperties, items, enum, const, "
             "anyOf, $ref, $defs)")
# Codex's apply_patch grammar (codex-rs/core/assets/tools/apply_patch.lark) and a line-level check of the same language.
APPLY_PATCH_LARK = ("start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?\n\n"
                    "hunk: add_hunk | delete_hunk | update_hunk\nadd_hunk: \"*** Add File: \" filename LF add_line+\n"
                    "delete_hunk: \"*** Delete File: \" filename LF\nupdate_hunk: \"*** Update File: \" filename LF "
                    "change_move? change?\n\nfilename: /(.+)/\nadd_line: \"+\" /(.*)/ LF -> line\n\n"
                    "change_move: \"*** Move to: \" filename LF\nchange: (change_context | change_line)+ eof_line?\n"
                    "change_context: (\"@@\" | \"@@ \" /(.+)/) LF\nchange_line: (\"+\" | \"-\" | \" \") /(.*)/ LF\n"
                    "eof_line: \"*** End of File\" LF\n\n%import common.LF\n")
PATCH_RE = re.compile(r"\*\*\* Begin Patch\n(?:"
                      r"\*\*\* Add File: [^\n]+\n(?:\+[^\n]*\n)+"
                      r"|\*\*\* Delete File: [^\n]+\n"
                      r"|\*\*\* Update File: [^\n]+\n(?:\*\*\* Move to: [^\n]+\n)?"
                      r"(?:(?:@@(?: [^\n]+)?\n|[-+ ][^\n]*\n)+(?:\*\*\* End of File\n)?)?"
                      r")+\*\*\* End Patch\n?")
APPLY_PATCH = {"type": "custom", "name": "apply_patch",
               "description": "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the "
                              "patch in JSON.",
               "format": {"type": "grammar", "syntax": "lark", "definition": APPLY_PATCH_LARK}}


def refused(server, path, body):
    status, out = server.request(path, body)
    assert status == 400, (status, out)
    return out["error"]["code"] + "|" + out["error"]["message"]


def output_text(resp):
    return "".join(c.get("text", "") for o in resp["output"] if o["type"] == "message" for c in o["content"])


def calls_of(resp, kind="function_call"):
    return [o for o in resp["output"] if o["type"] == kind]


def test_patch_check():
    assert PATCH_RE.fullmatch("*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch")
    assert PATCH_RE.fullmatch("*** Begin Patch\n*** Update File: a.py\n@@ def f\n-x\n+y\n*** End of File\n*** End Patch\n")
    assert not PATCH_RE.fullmatch("--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-x\n+y\n")
    assert not PATCH_RE.fullmatch("*** Begin Patch\n*** End Patch")


def test_completions_response_format(structured):
    for fmt in (schema_format(ANSWER), {"type": "json_object"}):
        out = structured.post("/v1/completions", {"prompt": "Q: What is the capital of France?\nA:", "temperature": 0,
                                                  "max_tokens": 200, "response_format": fmt})
        choice = out["choices"][0]
        assert choice["finish_reason"] == "stop", choice
        obj = json.loads(choice["text"])
        assert isinstance(obj, dict)
        if fmt["type"] == "json_schema":
            assert set(obj) == {"answer"} and isinstance(obj["answer"], str)


def test_completions_refusals(structured):
    body = {"prompt": "Q: hi\nA:", "max_tokens": 4}
    assert refused(structured, "/v1/completions", {**body, "response_format": schema_format({"type": "string", "pattern": "a"})}) == (
        "unsupported_json_schema|response_format.json_schema.schema: the keyword 'pattern' at \"\" " + SUPPORTED)
    assert refused(structured, "/v1/completions", {**body, "response_format": schema_format({"type": "text"})}) == (
        "invalid_json_schema|response_format.json_schema.schema: 'type' must be one of object, array, string, number, "
        "integer, boolean, null at \"\"")
    assert refused(structured, "/v1/completions", {**body, "response_format": {"type": "json_schema"}}) == (
        "invalid_request_error|response_format.json_schema.schema must be an object")


def test_chat_tool_choice_named(structured):
    """A greeting still yields exactly the named function."""
    out = structured.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Hello! How are you today?"}], "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "run_shell"}}, "temperature": 0, "max_tokens": 300,
        "chat_template_kwargs": NO_THINK})
    calls = out["choices"][0]["message"].get("tool_calls") or []
    assert calls and all(c["function"]["name"] == "run_shell" for c in calls)
    assert isinstance(json.loads(calls[0]["function"]["arguments"])["command"], str)


def test_chat_parallel_false_one_call(structured):
    out = structured.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Get the weather in Oslo, Paris, Rome and Lisbon."}], "tools": TOOLS,
        "tool_choice": "required", "parallel_tool_calls": False, "temperature": 0, "max_tokens": 400,
        "chat_template_kwargs": NO_THINK})
    calls = out["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) == 1 and out["choices"][0]["finish_reason"] == "tool_calls", calls


def test_chat_refusals(structured):
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "chat_template_kwargs": NO_THINK}
    assert refused(structured, "/v1/chat/completions", {**body, "tools": TOOLS, "tool_choice": {
        "type": "function", "function": {"name": "get_stock"}}}) == "invalid_tools|tool_choice: the function 'get_stock' is not in tools"
    assert refused(structured, "/v1/chat/completions", {**body, "response_format": schema_format({
        "type": "object", "properties": {"a": {"type": "string", "pattern": "x"}}})}) == (
        "unsupported_json_schema|response_format.json_schema.schema: the keyword 'pattern' at /properties/a " + SUPPORTED)


def test_responses_text_format(structured):
    out = structured.post("/v1/responses", {
        "input": "Tell me about Lisbon.", "reasoning": NO_REASONING, "temperature": 0, "max_output_tokens": 400,
        "text": {"format": {"type": "json_schema", "name": "city", "strict": True, "schema": CITY}}})
    check_city(output_text(out))
    out = structured.post("/v1/responses", {
        "input": "Say hello.", "reasoning": NO_REASONING, "temperature": 0, "max_output_tokens": 200,
        "text": {"format": {"type": "json_object"}}})
    assert isinstance(json.loads(output_text(out)), dict)


def test_responses_tool_choice(structured):
    base = {"input": "Hello! How are you today?", "tools": RESP_TOOLS, "reasoning": NO_REASONING, "temperature": 0,
            "max_output_tokens": 300}
    out = structured.post("/v1/responses", {**base, "tool_choice": "required"})
    assert calls_of(out) and not output_text(out).strip()
    out = structured.post("/v1/responses", {**base, "tool_choice": {"type": "function", "name": "run_shell"}})
    assert calls_of(out) and all(c["name"] == "run_shell" for c in calls_of(out))
    out = structured.post("/v1/responses", {**base, "input": "Get the weather in Oslo, Paris, Rome and Lisbon.",
                                            "tool_choice": "required", "parallel_tool_calls": False})
    assert len(calls_of(out)) == 1, out["output"]


def test_responses_apply_patch_enforced(structured):
    """Under tool_choice auto the apply_patch grammar is enforced: every call's input is a patch Codex can apply."""
    prompt = ("Create the file hello.txt containing the single line `hello` by calling the apply_patch tool. "
              "Call the tool now; do not explain.")
    for named in (False, True):
        body = {"input": prompt, "tools": [RESP_TOOLS[1], APPLY_PATCH], "reasoning": NO_REASONING, "temperature": 0,
                "max_output_tokens": 400}
        if named:
            body["tool_choice"] = {"type": "custom", "name": "apply_patch"}
        out = structured.post("/v1/responses", body)
        calls = calls_of(out, "custom_tool_call")
        assert calls, out["output"]
        for c in calls:
            assert c["name"] == "apply_patch" and PATCH_RE.fullmatch(c["input"]), c["input"]


def test_responses_refusals(structured):
    body = {"input": "hi", "max_output_tokens": 4, "reasoning": NO_REASONING}
    fmt = {"type": "json_schema", "name": "x", "schema": {"type": "string", "pattern": "a"}}
    assert refused(structured, "/v1/responses", {**body, "text": {"format": fmt}}) == (
        "unsupported_json_schema|text.format.schema: the keyword 'pattern' at \"\" " + SUPPORTED)
    assert refused(structured, "/v1/responses", {**body, "text": {"format": {"type": "json_schema", "name": "x"}}}) == (
        "invalid_request_error|text.format.schema must be an object")
    assert refused(structured, "/v1/responses", {**body, "tools": RESP_TOOLS, "tool_choice": {
        "type": "function", "name": "get_stock"}}) == "invalid_tools|tool_choice: the function 'get_stock' is not in tools"
    bad = {**APPLY_PATCH, "format": {"type": "grammar", "syntax": "ebnf", "definition": "root ::= \"a\""}}
    assert refused(structured, "/v1/responses", {**body, "tools": [RESP_TOOLS[0], bad]}) == (
        "unsupported_grammar|tools[1].format: the syntax 'ebnf' is not supported (supported: lark, regex)")
    bad["format"] = {"type": "grammar", "syntax": "lark", "definition": "start: \"a\"i\n"}
    assert refused(structured, "/v1/responses", {**body, "tools": [RESP_TOOLS[0], bad]}).startswith(
        "unsupported_grammar|tools[1].format: ")
    bad["format"] = {"type": "grammar", "syntax": "lark", "definition": ""}
    assert refused(structured, "/v1/responses", {**body, "tools": [bad]}) == (
        "invalid_grammar|tools[0].format: the definition is missing or empty")
    bad["format"] = {"type": "grammar", "syntax": "lark", "definition": "start: missing_rule\n"}
    assert refused(structured, "/v1/responses", {**body, "tools": [bad]}).startswith("invalid_grammar|")
