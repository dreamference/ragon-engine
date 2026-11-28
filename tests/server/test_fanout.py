# BATCHING P4d: `n`, a request's choices. The first prefills, the others start as forks of its session.
# Run against a server with the page pool (the default; not --kv-flat) and at least eight sessions; a smaller one still answers (forks queue).

import threading

import pytest

from conftest import stream_chat_parts
from test_regressions import NO_THINK, code_text, evict, greedy_chat, resident_sessions


def question(tag, lines=40):
    return [{"role": "system", "content": f"Review {tag}. Answer in one short line.\n" + code_text(lines)},
            {"role": "user", "content": "What does f3 return for x = 2?"}]


def texts(resp):
    return [c["message"].get("content") or "" for c in resp["choices"]]


@pytest.mark.parametrize("bad", [0, -1, 129, "2", 1.5, True])
def test_n_out_of_range_is_a_400(server, bad):
    status, out = server.request("/v1/chat/completions", {"messages": question("bad", 1), "max_tokens": 1, "n": bad})
    assert status == 400, out


def test_n_null_is_one_choice(server):
    resp = greedy_chat(server, question("null", 1), max_tokens=2, n=None)
    assert [c["index"] for c in resp["choices"]] == [0]


@pytest.mark.parametrize("n", [2, 4, 12])
def test_greedy_choices_are_the_one_answer(server, n):
    """Speculative pairs at n = 2 and 4, plain rows past --fanout-plain-above: the same tokens as one request."""
    resp = greedy_chat(server, question(f"greedy{n}b"), max_tokens=32, n=n)
    alone = greedy_chat(server, question(f"greedy{n}b"), max_tokens=32)
    assert [c["index"] for c in resp["choices"]] == list(range(n))
    assert texts(resp) == texts(alone) * n
    usage = resp["usage"]
    assert usage["prompt_tokens"] == alone["usage"]["prompt_tokens"], "the prompt is counted once"
    assert usage["completion_tokens"] == n * alone["usage"]["completion_tokens"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_sampled_choices_differ(server):
    body = {"messages": question("sampled"), "max_tokens": 24, "temperature": 1.0, "top_k": 40, "seed": 5, "n": 6,
            "chat_template_kwargs": NO_THINK}
    assert len(set(texts(server.post("/v1/chat/completions", body)))) >= 2


def test_a_stream_carries_every_choice(server):
    alone = greedy_chat(server, question("stream"), max_tokens=24)
    body = {"messages": question("stream"), "max_tokens": 24, "temperature": 0, "n": 4, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": NO_THINK}
    status, events = server.stream("/v1/chat/completions", body)
    assert status == 200
    assert events[-1]["data"] == "[DONE]" and sum(e["data"] == "[DONE]" for e in events) == 1
    by_choice = {i: [] for i in range(4)}
    for e in events[:-1]:
        for ch in e["data"].get("choices") or []:
            by_choice[ch["index"]].append(e)
    for i in range(4):
        content, _, _, finishes, _, _ = stream_chat_parts(
            [{"data": {"choices": [c for c in e["data"]["choices"] if c["index"] == i]}} for e in by_choice[i]])
        assert content == texts(alone)[0]
        assert len(finishes) == 1
    usage = [e["data"]["usage"] for e in events[:-1] if e["data"].get("usage")]
    assert len(usage) == 1 and usage[0]["completion_tokens"] == 4 * alone["usage"]["completion_tokens"]


def test_completions_take_n(server):
    body = {"prompt": code_text(40), "max_tokens": 8, "temperature": 0, "n": 3}
    resp = server.post("/v1/completions", body)
    assert [c["index"] for c in resp["choices"]] == [0, 1, 2]
    assert len({c["text"] for c in resp["choices"]}) == 1


def test_more_choices_than_the_server_holds_queue(server):
    """Past the sessions, --batch or the 32 rows of a pass a fork waits and runs later: every choice is answered."""
    resp = greedy_chat(server, question("past"), max_tokens=6, n=40)
    assert [c["index"] for c in resp["choices"]] == list(range(40))
    assert len(set(texts(resp))) == 1
    assert all(c["finish_reason"] for c in resp["choices"])


def test_a_client_that_leaves_ends_every_choice(server):
    body = {"messages": question("leave"), "max_tokens": 2000, "temperature": 0, "n": 4, "stream": True,
            "chat_template_kwargs": NO_THINK}
    r = server.open_stream("/v1/chat/completions", body)
    for raw in r:
        if raw.startswith(b"data: "):
            break
    r.close()
    assert greedy_chat(server, question("after"), max_tokens=4)["choices"][0]["finish_reason"]


def test_two_requests_with_choices_at_once(server):
    out = {}

    def ask(tag):
        out[tag] = greedy_chat(server, question(tag), max_tokens=12, n=4)

    threads = [threading.Thread(target=ask, args=(t,)) for t in ("twoa", "twob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for tag in ("twoa", "twob"):
        assert len(set(texts(out[tag]))) == 1 and len(out[tag]["choices"]) == 4


LONG_IDS = [1000 + i % 5000 for i in range(20000)]  # past the checkpoint store's chunk multiples (eight, 2048 apart)


def completion(server, ids, max_tokens=1):
    status, out = server.request(
        "/v1/completions", {"prompt": ids, "max_tokens": max_tokens, "temperature": 0}
    )
    assert status == 200, out
    return out


def test_the_second_fork_resumes_where_the_first_one_did(server):
    """The first follow-up rewrites the session's end state: the point it left stays in the store for the next one."""
    completion(server, LONG_IDS)
    completion(server, LONG_IDS + [7001, 7002, 7003])
    second = completion(server, LONG_IDS + [7004, 7005, 7006])
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] >= len(LONG_IDS)


def test_many_forks_with_their_own_questions_answer_as_alone(server):
    """Past --fanout-plain-above live requests the question prefills run ahead of decoding; the answers do not change."""
    ids = LONG_IDS[:4096]
    questions = [ids + [7100 + 3 * i, 7101 + 3 * i, 7102 + 3 * i] for i in range(12)]
    alone = [completion(server, q, 16)["choices"][0]["text"] for q in questions]
    together = [None] * len(questions)

    def one(i):
        together[i] = completion(server, questions[i], 16)["choices"][0]["text"]

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(questions))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert together == alone


def metric(server, name):
    """A counter or gauge from /metrics, None when the server does not export it."""
    status, text = server.request("/metrics")
    for line in text.splitlines() if status == 200 else []:
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return None


def test_other_threads_that_share_a_session_prefix_fork_it(server):
    """TRUE_FORK, fork admission: requests of other threads (the thread-id header, as Codex sends it) that extend an
    idle session's prompt start as forks of that session, at once and side by side: none re-reads the prefix, none
    moves the session on under the others, the session's own thread still continues it, and each answers as alone."""
    status, models = server.request("/v1/models")
    if "session-fork" not in (models["data"][0].get("features") or []):
        pytest.skip("the server has no fork admission (no 'session-fork' feature in /v1/models)")
    sessions = resident_sessions(server)
    if sessions < 3:
        pytest.skip(f"{sessions} sessions: forks need slots beside their parent's")
    k = min(4, sessions - 1)
    ids = LONG_IDS[:12000]
    branches = [ids + [7200 + 5 * i, 7201 + 5 * i, 7202 + 5 * i] for i in range(k)]
    alone = []
    for b in branches:
        evict(server)
        alone.append(completion(server, b, 16)["choices"][0]["text"])
    evict(server)
    parent = server.post("/v1/completions", {"prompt": ids, "max_tokens": 8, "temperature": 0},
                         headers={"thread-id": "fork-parent"})
    assert parent["choices"][0]["finish_reason"]
    before = metric(server, "ragon:session_forks_total")
    assert before is not None, "no ragon:session_forks_total in /metrics"
    out = [None] * k

    def one(i):
        out[i] = server.post("/v1/completions", {"prompt": branches[i], "max_tokens": 16, "temperature": 0},
                             headers={"thread-id": f"fork-branch-{i}"})

    threads = [threading.Thread(target=one, args=(i,)) for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert metric(server, "ragon:session_forks_total") - before >= k, "a branch did not start as a fork"
    for i in range(k):
        assert out[i]["usage"]["prompt_tokens_details"]["cached_tokens"] >= len(ids), f"branch {i} re-read the prefix"
    assert [o["choices"][0]["text"] for o in out] == alone
    # The parent's thread continues its session, which no branch moved on.
    again = server.post("/v1/completions", {"prompt": ids + [7300, 7301], "max_tokens": 4, "temperature": 0},
                        headers={"thread-id": "fork-parent"})
    assert again["usage"]["prompt_tokens_details"]["cached_tokens"] >= len(ids)
