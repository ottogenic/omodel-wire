#!/usr/bin/env python3
"""Offline tests for utils/omw_loom.py (the loom conductor).

A scripted fake OpenCode server (stdlib http.server) plays the workers: each
agent has a FIFO of canned Return-Contract replies, and every request is
recorded so tests can assert on parentID, session reuse, agent routing, and
model overrides. No network beyond 127.0.0.1, no $HOME writes (ledger in tmp).

Run: python3 -m unittest test_omw_loom -v
"""

import http.server
import importlib.util
import json
import os
import tempfile
import threading
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "omw_loom", os.path.join(_HERE, "utils", "omw_loom.py"))
loom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loom)


MODEL_ERROR = "___MODEL_ERROR___"   # sentinel: fake server answers with an API error


def reply(status, result="ok", evidence="checks ran", extra="", body=""):
    """A well-formed Return-Contract worker reply (loom v3: no NEXT STEPS field).
    `body` is the optional full deliverable ABOVE the contract block."""
    lines = [f"RESULT: {result}", f"EVIDENCE: {evidence}", extra,
             f"STATUS: {status}"]
    block = "\n".join(l for l in lines if l)
    return f"{body}\n\n{block}" if body else block


class FakeOpenCode:
    """Scripted server: responses[agent] is a FIFO of reply texts."""

    def __init__(self):
        self.responses = {}
        self.requests = []          # (method, path, body) in arrival order
        self.sessions = []          # created session dicts
        self.aborted = []
        self.deleted = []
        self.sse_events = []        # scripted /event payloads (SSE)
        self.sse_delay = 0.05       # seconds before the /event stream emits
        self.message_delay = 0.0    # seconds before a message POST answers
        self._n = 0
        self._lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: A003 -- silence
                pass

            def _body(self):
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length).decode() if length else ""
                return json.loads(raw) if raw else {}

            def _send(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                with outer._lock:
                    outer.requests.append(("GET", self.path, {}))
                if self.path == "/event":
                    # Minimal SSE: emit the scripted events once, then close.
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    import time as _t
                    _t.sleep(outer.sse_delay)
                    for ev in outer.sse_events:
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                    _t.sleep(0.3)
                    return
                self._send(outer.sessions)

            def do_POST(self):
                body = self._body()
                with outer._lock:
                    outer.requests.append(("POST", self.path, body))
                if self.path.startswith("/session?") or self.path == "/session":
                    with outer._lock:
                        outer._n += 1
                        sess = {"id": f"ses_{outer._n}",
                                "parentID": body.get("parentID"),
                                "title": body.get("title")}
                        outer.sessions.append(sess)
                    self._send(sess)
                    return
                if self.path.endswith("/abort"):
                    with outer._lock:
                        outer.aborted.append(self.path.split("/")[2])
                    self._send({})
                    return
                # /session/{id}/message
                if outer.message_delay:
                    import time as _t
                    _t.sleep(outer.message_delay)
                sid = self.path.split("/")[2]
                agent = body.get("agent", "?")
                with outer._lock:
                    queue = outer.responses.setdefault(agent, [])
                    text = queue.pop(0) if queue else reply("DONE", "default")
                if text == MODEL_ERROR:
                    self._send({"info": {"role": "assistant",
                                         "error": {"name": "APIError",
                                                   "data": {"message": "model gone"}}},
                                "parts": []})
                    return
                self._send({"info": {"role": "assistant"},
                            "parts": [{"type": "text", "text": text}]})

            def do_DELETE(self):
                with outer._lock:
                    outer.requests.append(("DELETE", self.path, {}))
                    outer.deleted.append(self.path.split("/")[2])
                self._send({})

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def prompts(self, agent=None):
        """[(session_id, agent, text, model)] for every message POST, in order."""
        out = []
        for method, path, body in self.requests:
            if method == "POST" and "/message" in path:
                sid = path.split("/")[2]
                if agent is None or body.get("agent") == agent:
                    text = body["parts"][0]["text"]
                    out.append((sid, body.get("agent"), text, body.get("model")))
        return out

    def created(self):
        return list(self.sessions)


class LoomCase(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.server = FakeOpenCode()
        # ignore_cleanup_errors: on Windows a WAL sidecar can outlive the close.
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dbfile = os.path.join(self.tmp.name, "loom.db")
        self.led = loom.Ledger(self.dbfile)
        self.cfg = dict(loom.DEFAULTS, step_timeout=30)

    def tearDown(self):
        self.led.close()
        self.server.stop()
        self.tmp.cleanup()

    def make_loom(self, risk="simple", ranking=None, worker_model=None, goal="do the thing"):
        tp = loom.Transport(self.server.url, retries=1)
        job_id = self.led.create_job(dir=self.tmp.name, server_url=self.server.url,
                                     parent_session="ses_parent", goal=goal,
                                     risk=risk, worker_model=worker_model)
        lm = loom.Loom(self.led, tp, job_id, cfg=self.cfg,
                       model_ranking=ranking or {}, out=open(os.devnull, "w"))
        return lm, job_id

    # -- contract parser ----------------------------------------------------------

    def test_parse_contract_full(self):
        p = loom.parse_contract(reply("DONE", "changed a.py:12", "pytest: 3 passed"))
        self.assertEqual(p["status"], "DONE")
        self.assertIn("a.py:12", p["result"])
        self.assertIn("3 passed", p["evidence"])
        # loom v3 dropped the NEXT STEPS field entirely -- routing is the
        # conductor's job, never parsed from the worker's reply.
        self.assertEqual(set(p), {"status", "result", "evidence",
                                  "research", "findings", "clean"})

    def test_parse_contract_tolerates_backticks_and_case(self):
        p = loom.parse_contract("stuff\n`STATUS`: `blocked`\ntrailing text")
        self.assertEqual(p["status"], "BLOCKED")

    def test_parse_contract_malformed(self):
        self.assertIsNone(loom.parse_contract("I did some things, all good!")["status"])

    def test_parse_result_stops_at_next_contract_label(self):
        # Section lookahead is RESULT|EVIDENCE|STATUS only.
        p = loom.parse_contract("RESULT: the summary\nEVIDENCE: cmd -> ok\nSTATUS: DONE")
        self.assertEqual(p["result"], "the summary")
        self.assertEqual(p["evidence"], "cmd -> ok")

    def test_parse_research_requests(self):
        text = ("RESULT: need facts\nEVIDENCE: none\nSTATUS: NEEDS_RESEARCH\n"
                "RESEARCH REQUEST:\n1. What DB engine is used?\n2. Which auth lib?\n")
        p = loom.parse_contract(text)
        self.assertEqual(p["research"],
                         ["What DB engine is used?", "Which auth lib?"])

    def test_parse_findings_and_clean(self):
        text = ("RESULT:\nFINDING 1: a.py:3 missing null check | PASS CONDITION: check added\n"
                "FINDING 2: b.py:9 secret logged | PASS CONDITION: removed\n"
                "EVIDENCE: read diff\nSTATUS: DONE")
        p = loom.parse_contract(text)
        self.assertEqual([n for n, _ in p["findings"]], [1, 2])
        self.assertIn("null check", p["findings"][0][1])
        self.assertTrue(loom.parse_contract("No blocking findings.\nSTATUS: DONE")["clean"])

    def test_strip_contract_tail_returns_body_above_final_result(self):
        text = ("## The full plan\n\n1. do this\n2. do that\n\n"
                "RESULT: a short synopsis\nEVIDENCE: read the repo\nSTATUS: DONE")
        self.assertEqual(loom.strip_contract_tail(text),
                         "## The full plan\n\n1. do this\n2. do that")

    def test_strip_contract_tail_no_result_line_returns_whole_text(self):
        self.assertEqual(loom.strip_contract_tail("just prose, no contract"),
                         "just prose, no contract")

    def test_strip_contract_tail_contract_only_reply_returns_whole_text(self):
        text = "RESULT: all of it\nEVIDENCE: ran x\nSTATUS: DONE"
        self.assertEqual(loom.strip_contract_tail(text), text)

    def test_strip_contract_tail_multiple_result_lines_cuts_at_last(self):
        text = ("intro\nRESULT: an inline echo in the body\nmore body\n"
                "RESULT: final summary\nEVIDENCE: e\nSTATUS: DONE")
        self.assertEqual(loom.strip_contract_tail(text),
                         "intro\nRESULT: an inline echo in the body\nmore body")

    def test_strip_contract_tail_empty(self):
        self.assertEqual(loom.strip_contract_tail(""), "")
        self.assertEqual(loom.strip_contract_tail(None), "")

    # -- happy path -----------------------------------------------------------------

    def test_simple_happy_path(self):
        self.server.responses = {
            "agent-code": [reply("DONE", "implemented x")],
            "agent-test": [reply("DONE", "all pass", "34/34 PASS")],
            "agent-review": ["No blocking findings.\n" + reply("DONE", "clean review")],
        }
        lm, job_id = self.make_loom(risk="simple")
        lm.run()
        job = self.led.job(job_id)
        self.assertEqual(job["status"], "done")
        # plan skipped, order: code -> test -> review
        agents = [a for _, a, _, _ in self.server.prompts()]
        self.assertEqual(agents, ["agent-code", "agent-test", "agent-review"])
        # every child parented under the loom agent session, task-tool title format
        for s in self.server.created():
            self.assertEqual(s["parentID"], "ses_parent")
            self.assertIn("subagent)", s["title"])
        self.assertIn("LOOM JOB COMPLETE", job["report"])

    def test_plan_phase_with_parallel_research(self):
        self.server.responses = {
            "agent-architect": [
                reply("NEEDS_RESEARCH", "need facts", extra=(
                    "RESEARCH REQUEST:\n1. Which framework?\n2. Which test runner?")),
                reply("DONE", "PLAN: 1. do it. CRITERIA: works."),
            ],
            "agent-research": [reply("DONE", "framework: flask"),
                               reply("DONE", "runner: pytest")],
            "agent-code": [reply("DONE", "implemented")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom(risk="medium")
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        # two research children spawned
        research = [s for s in self.server.created() if "agent-research" in s["title"]]
        self.assertEqual(len(research), 2)
        # architect resumed in the SAME session for round 2
        arch = self.server.prompts("agent-architect")
        self.assertEqual(len(arch), 2)
        self.assertEqual(arch[0][0], arch[1][0])
        self.assertIn("Research results", arch[1][2])
        # the plan reached agent-code's packet
        self.assertIn("PLAN: 1. do it", self.server.prompts("agent-code")[0][2])
        # the fan-out reported per-completion progress for the live card
        details = [e["detail"] for e in self.led.events(job_id) if e["kind"] == "research"]
        self.assertIn("answered 2/2", details)

    # -- packet composition (loom v3: instructions + contract injected per dispatch) --

    def _write(self, *parts, text):
        path = os.path.join(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def _skill_dirs(self):
        """(global loom_dir, repo .loom/skills dir) under the job dir."""
        gdir = os.path.join(self.tmp.name, "gskills")
        rdir = os.path.join(self.tmp.name, ".loom", "skills")
        os.makedirs(gdir, exist_ok=True)
        os.makedirs(rdir, exist_ok=True)
        return gdir, rdir

    def test_compose_packet_override_beats_default_and_local(self):
        gdir, rdir = self._skill_dirs()
        self._write(gdir, "agent-code-instructions-default.md", text="DEFAULT INSTRUCTIONS")
        self._write(rdir, "agent-code-instructions-local.md", text="LOCAL OVERLAY")
        self._write(rdir, "agent-code-instructions-override.md", text="OVERRIDE TEXT")
        self._write(gdir, "agent-code-contract.md", text="ROLE CONTRACT TEXT")
        self.cfg["loom_dir"] = gdir
        lm, _ = self.make_loom()
        packet, files = lm._compose_packet("agent-code", "TASK PROMPT")
        self.assertTrue(packet.startswith("TASK PROMPT"))
        self.assertIn("OVERRIDE TEXT", packet)
        # override is EXCLUSIVE: default and local must not leak in beside it
        self.assertNotIn("DEFAULT INSTRUCTIONS", packet)
        self.assertNotIn("LOCAL OVERLAY", packet)
        # the contract sits at the tail (most-recent instruction wins)
        self.assertTrue(packet.endswith("ROLE CONTRACT TEXT"))
        self.assertTrue(any(f.startswith("override:") for f in files), files)
        self.assertIn("contract", files)

    def test_compose_packet_default_plus_local_when_no_override(self):
        gdir, rdir = self._skill_dirs()
        self._write(gdir, "agent-code-instructions-default.md", text="DEFAULT INSTRUCTIONS")
        self._write(rdir, "agent-code-instructions-local.md", text="LOCAL OVERLAY")
        self._write(gdir, "agent-code-contract.md", text="ROLE CONTRACT TEXT")
        self.cfg["loom_dir"] = gdir
        lm, _ = self.make_loom()
        packet, files = lm._compose_packet("agent-code", "TASK PROMPT")
        # order: task, default instructions, repo overlay, contract last
        self.assertLess(packet.index("TASK PROMPT"), packet.index("DEFAULT INSTRUCTIONS"))
        self.assertLess(packet.index("DEFAULT INSTRUCTIONS"), packet.index("LOCAL OVERLAY"))
        self.assertLess(packet.index("LOCAL OVERLAY"), packet.index("ROLE CONTRACT TEXT"))
        self.assertTrue(packet.endswith("ROLE CONTRACT TEXT"))
        self.assertTrue(any(f.startswith("default:") for f in files), files)
        self.assertTrue(any(f.startswith("local:") for f in files), files)

    def test_compose_packet_missing_files_is_bare_prompt(self):
        gdir, _ = self._skill_dirs()   # both dirs exist but are empty
        self.cfg["loom_dir"] = gdir
        lm, _ = self.make_loom()
        packet, files = lm._compose_packet("agent-code", "TASK PROMPT")
        self.assertEqual(packet, "TASK PROMPT")
        self.assertEqual(files, [])

    def test_compose_packet_contract_falls_back_to_injected_text(self):
        # No per-role contract file -> cfg["status_contract"] (wire-injected) is used.
        gdir, _ = self._skill_dirs()
        self.cfg["loom_dir"] = gdir
        self.cfg["status_contract"] = "FALLBACK CONTRACT"
        lm, _ = self.make_loom()
        packet, files = lm._compose_packet("agent-code", "TASK PROMPT")
        self.assertTrue(packet.endswith("FALLBACK CONTRACT"))
        self.assertIn("contract", files)
        # ...but a per-role contract file WINS over the fallback
        self._write(gdir, "agent-code-contract.md", text="ROLE CONTRACT TEXT")
        packet2, _ = lm._compose_packet("agent-code", "TASK PROMPT")
        self.assertTrue(packet2.endswith("ROLE CONTRACT TEXT"))
        self.assertNotIn("FALLBACK CONTRACT", packet2)

    def test_fresh_dispatch_composes_packet_resume_does_not(self):
        # Fresh sessions get [prompt + instructions + contract] and a "compose"
        # ledger event; resumed sessions get the bare prompt (already in-context).
        gdir, _ = self._skill_dirs()
        self._write(gdir, "agent-code-instructions-default.md", text="DEFAULT INSTRUCTIONS")
        self.cfg["loom_dir"] = gdir
        self.cfg["status_contract"] = "FALLBACK CONTRACT"
        self.server.responses = {
            "agent-code": [reply("DONE", "implemented"),
                           reply("DONE", "fixed the assert")],
            "agent-test": [reply("DONE", "1 failure", "FAIL test_x: boom"),
                           reply("DONE", "all pass", "34/34 PASS")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        code = self.server.prompts("agent-code")
        self.assertIn("DEFAULT INSTRUCTIONS", code[0][2])
        self.assertTrue(code[0][2].endswith("FALLBACK CONTRACT"))
        # the resumed fix prompt is NOT re-composed (no wasted tokens)
        self.assertNotIn("DEFAULT INSTRUCTIONS", code[1][2])
        self.assertNotIn("FALLBACK CONTRACT", code[1][2])
        # one compose event per fresh session, named per role
        composes = [e["detail"] for e in self.led.events(job_id) if e["kind"] == "compose"]
        self.assertEqual(len(composes), len(self.server.created()))
        self.assertTrue(any(c.startswith("agent-code:") for c in composes), composes)

    def test_plan_handoff_carries_full_deliverable_not_result_summary(self):
        # Regression (job 60): the architect's 3.5k-char plan was collapsed to its
        # 624-char RESULT synopsis. The stored plan must be the body ABOVE the
        # contract block, with the RESULT summary as fallback only.
        plan_body = "## Full plan\n1. step one\n2. step two\nCRITERIA: it works"
        self.server.responses = {
            "agent-architect": [reply("DONE", "short synopsis", body=plan_body)],
            "agent-code": [reply("DONE", "implemented")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom(risk="medium")
        lm.run()
        job = self.led.job(job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["plan"], plan_body)
        # the FULL plan (not the synopsis) reaches agent-code's packet
        code_packet = self.server.prompts("agent-code")[0][2]
        self.assertIn("2. step two", code_packet)

    def test_nudge_merges_original_body_with_nudged_contract(self):
        # A malformed (no STATUS) plan reply is nudged for ONLY the contract block;
        # the deliverable from the first reply and the contract from the second are
        # merged, so the plan handoff keeps the full artifact.
        self.cfg["status_contract"] = ("CONTRACT-MARKER\nRESULT: <r>\n"
                                       "EVIDENCE: <e>\nSTATUS: <s>")
        plan_body = "## The real plan\nstep A\nstep B"
        self.server.responses = {
            "agent-architect": [plan_body,                    # no STATUS -> nudge
                                reply("DONE", "nudged synopsis")],
            "agent-code": [reply("DONE", "implemented")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom(risk="medium")
        lm.run()
        job = self.led.job(job_id)
        self.assertEqual(job["status"], "done")
        arch = self.server.prompts("agent-architect")
        self.assertEqual(len(arch), 2)
        self.assertEqual(arch[0][0], arch[1][0], "nudge stays in the same session")
        # the nudge asks for ONLY the contract block, quoting the contract text,
        # and explicitly forbids regenerating the delivered work
        self.assertIn("ONLY the", arch[1][2])
        self.assertIn("CONTRACT-MARKER", arch[1][2])
        self.assertIn("do NOT repeat or rewrite it", arch[1][2])
        # merged handoff: the plan is the original body, not the nudged synopsis
        self.assertEqual(job["plan"], plan_body)

    def test_model_for_falls_back_to_role_models(self):
        # No job-wide override, no ranking bump -> the wire-injected per-role model
        # is passed EXPLICITLY on every dispatch (never server-side resolution).
        self.cfg["role_models"] = {"agent-code": "prov/mod"}
        self.server.responses = {
            "agent-code": [reply("DONE")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        code_models = [mdl for _, _, _, mdl in self.server.prompts("agent-code")]
        self.assertEqual(code_models, [{"providerID": "prov", "modelID": "mod"}])
        # roles without an entry send no explicit model
        test_models = [mdl for _, _, _, mdl in self.server.prompts("agent-test")]
        self.assertEqual(test_models, [None])

    def test_dispatch_kwargs_fold_into_loom_config(self):
        # omodel-wire injects the contract, per-role models, and the loom files dir
        # via dispatch(...); loom_config() must fold all three into every job cfg.
        import contextlib
        import io
        saved = (loom._STATUS_CONTRACT, loom._ROLE_MODELS, loom._LOOM_DIR)

        class Args:
            loom_cmd = "status"
            job = None
            db = self.dbfile

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = loom.dispatch(Args(), role_models={"agent-code": "p/m"},
                                   loom_dir="/cfg/loom/skills",
                                   status_contract="THE CONTRACT")
            self.assertEqual(rc, 0)
            cfg = loom.loom_config()
            self.assertEqual(cfg["role_models"], {"agent-code": "p/m"})
            self.assertEqual(cfg["loom_dir"], "/cfg/loom/skills")
            self.assertEqual(cfg["status_contract"], "THE CONTRACT")
        finally:
            loom._STATUS_CONTRACT, loom._ROLE_MODELS, loom._LOOM_DIR = saved

    # -- failure loops ---------------------------------------------------------------

    def test_test_failure_routes_back_to_same_code_session(self):
        self.server.responses = {
            "agent-code": [reply("DONE", "implemented"),
                           reply("DONE", "fixed the assert")],
            "agent-test": [reply("DONE", "1 failure", "FAIL test_x: boom"),
                           reply("DONE", "all pass", "34/34 PASS")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        code = self.server.prompts("agent-code")
        self.assertEqual(len(code), 2)
        self.assertEqual(code[0][0], code[1][0], "fix must resume the SAME code session")
        self.assertIn("FAIL test_x", code[1][2])
        tests = self.server.prompts("agent-test")
        self.assertEqual(tests[0][0], tests[1][0], "retest must resume the SAME test session")

    def test_review_findings_fixed_one_at_a_time_in_new_sessions(self):
        finding_reply = ("FINDING 1: a.py:3 null check | PASS CONDITION: added\n"
                         "FINDING 2: b.py:9 secret log | PASS CONDITION: removed\n"
                         "STATUS: DONE")
        second = ("FINDING 2: b.py:9 secret log | PASS CONDITION: removed\n"
                  "STATUS: DONE")
        self.server.responses = {
            "agent-code": [reply("DONE", "implemented"),
                           reply("DONE", "fixed finding 1"),
                           reply("DONE", "fixed finding 2")],
            "agent-test": [reply("DONE"),
                           reply("DONE", "recheck 1 pass"),
                           reply("DONE", "recheck 2 pass")],
            "agent-review": [finding_reply, second, "No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        reviews = self.server.prompts("agent-review")
        self.assertEqual(len(reviews), 3)
        self.assertEqual({r[0] for r in reviews}, {reviews[0][0]},
                         "all reviews share ONE session")
        self.assertIn("Finding 1 was fixed", reviews[1][2])
        code = self.server.prompts("agent-code")
        self.assertEqual(len(code), 3)
        self.assertEqual(len({c[0] for c in code}), 3,
                         "each finding gets a NEW agent-code session")
        # one at a time: fix-1 dispatched before re-review round 2, fix-2 after it
        order = [(a, t) for _, a, t, _ in self.server.prompts()]
        fixes = [i for i, (a, t) in enumerate(order)
                 if a == "agent-code" and t.startswith("Fix exactly this review finding")]
        rereview = next(i for i, (a, t) in enumerate(order)
                        if a == "agent-review" and "Finding 1 was fixed" in t)
        self.assertEqual(len(fixes), 2)
        self.assertLess(fixes[0], rereview)
        self.assertLess(rereview, fixes[1])
        self.assertIn("null check", order[fixes[0]][1])
        self.assertIn("secret log", order[fixes[1]][1])

    def test_blocked_code_gets_architect_diagnosis_then_resumes(self):
        self.server.responses = {
            "agent-code": [reply("BLOCKED", "cannot find api", "tried x, error y"),
                           reply("DONE", "applied correction")],
            "agent-architect": [reply("DONE", "correction: use client.v2")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        arch = self.server.prompts("agent-architect")
        self.assertEqual(len(arch), 1)
        self.assertIn("BLOCKED REPORT", arch[0][2])
        code = self.server.prompts("agent-code")
        self.assertEqual(code[0][0], code[1][0], "correction resumes the same session")
        self.assertIn("use client.v2", code[1][2])

    def test_antispin_two_stale_continues_become_blocked(self):
        self.server.responses = {
            "agent-code": [reply("CONTINUE", "working", "same evidence"),
                           reply("CONTINUE", "working", "same evidence"),
                           reply("DONE", "after correction")],
            "agent-architect": [reply("DONE", "correction: do z")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        # architect got the (synthesized) blocked report -- the anti-spin fired
        self.assertEqual(len(self.server.prompts("agent-architect")), 1)

    def test_malformed_reply_gets_one_nudge_then_proceeds(self):
        self.server.responses = {
            "agent-code": ["did stuff, looks fine!",       # no STATUS -> nudge
                           reply("DONE", "proper contract")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        code = self.server.prompts("agent-code")
        self.assertEqual(len(code), 2)
        self.assertEqual(code[0][0], code[1][0])
        self.assertIn("Return Contract", code[1][2])

    # -- escalation ladder -------------------------------------------------------------

    def test_escalation_bumps_model_then_pauses_resumably(self):
        blocked = reply("BLOCKED", "stuck", "no idea")
        nodiag = reply("BLOCKED", "cannot diagnose", "unknown")
        self.server.responses = {
            "agent-code": [blocked, blocked, blocked, blocked],
            "agent-architect": [nodiag, nodiag, nodiag, nodiag],
        }
        ranking = {"agent-code": ["local/small", "big/frontier-model"]}
        lm, job_id = self.make_loom(ranking=ranking)
        lm.run()
        job = self.led.job(job_id)
        self.assertEqual(job["status"], "paused")
        self.assertIn("resume", job["report"])
        # ladder rung 3 dispatched agent-code on the next-ranked model
        models = [m for _, _, _, m in self.server.prompts("agent-code") if m]
        self.assertIn({"providerID": "big", "modelID": "frontier-model"}, models)
        # sessions: implement got fresh sessions on escalation rungs
        code_sessions = {sid for sid, _, _, _ in self.server.prompts("agent-code")}
        self.assertGreaterEqual(len(code_sessions), 2)

        # now the human fixes the world: resume completes the job
        self.server.responses = {
            "agent-code": [reply("DONE", "unblocked after human note")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        self.led.update_job(job_id, status="running")
        self.led.add_note(job_id, "use the staging endpoint")
        tp = loom.Transport(self.server.url, retries=1)
        lm2 = loom.Loom(self.led, tp, job_id, cfg=self.cfg, out=open(os.devnull, "w"))
        lm2.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        # the operator note reached the next dispatch
        resumed = self.server.prompts("agent-code")[-1][2]
        self.assertIn("staging endpoint", resumed)

    def test_worker_model_override_applies_to_every_dispatch(self):
        self.server.responses = {
            "agent-code": [reply("DONE")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom(worker_model="google/gemini-cheap")
        lm.run()
        for _, agent, _, model in self.server.prompts():
            self.assertEqual(model, {"providerID": "google", "modelID": "gemini-cheap"},
                             f"{agent} must run on the override model")

    # -- live activity narration -----------------------------------------------------------

    def test_describe_part_variants(self):
        d = loom.describe_part
        self.assertEqual(d({"type": "reasoning"}), "thinking...")
        self.assertEqual(
            d({"type": "tool", "tool": "bash",
               "state": {"status": "running", "input": {"command": "python3  -m pytest -x"}}}),
            "$ python3 -m pytest -x")
        self.assertEqual(
            d({"type": "tool", "tool": "todowrite",
               "state": {"status": "running", "input": {"todos": [
                   {"status": "completed"}, {"status": "pending"}, {"status": "completed"}]}}}),
            "todos 2/3")
        self.assertEqual(
            d({"type": "tool", "tool": "edit",
               "state": {"status": "running", "input": {"filePath": "/a/b/hello.py"}}}),
            "edit hello.py")
        self.assertIsNone(d({"type": "text"}))
        self.assertIsNone(d({"type": "tool", "tool": "bash", "state": {"status": "error"}}))

    def test_activity_watcher_narrates_active_worker(self):
        # While agent-code's prompt is in flight, a part event for ITS session must
        # surface as an ephemeral activity line (stdout only, not the ledger).
        import io
        self.server.message_delay = 0.6
        self.server.sse_events = [{
            "type": "message.part.updated",
            "properties": {"part": {"sessionID": "ses_1", "type": "tool", "tool": "bash",
                                    "state": {"status": "running",
                                              "input": {"command": "pytest -q"}}}},
        }]
        self.server.responses = {
            "agent-code": [reply("DONE")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        out = io.StringIO()
        tp = loom.Transport(self.server.url, retries=1)
        job_id = self.led.create_job(dir=self.tmp.name, server_url=self.server.url,
                                     parent_session="ses_parent", goal="g", risk="simple")
        lm = loom.Loom(self.led, tp, job_id, cfg=self.cfg, json_events=True, out=out)
        lm.run()
        lines = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
        acts = [l for l in lines if l["type"] == "activity"]
        self.assertTrue(acts, "no activity events were narrated")
        self.assertIn("$ pytest -q", acts[0]["detail"])
        self.assertIn("code", acts[0]["detail"])
        # ledger stays transition-only
        kinds = {e["kind"] for e in self.led.events(job_id)}
        self.assertNotIn("activity", kinds)
        # dispatch events carry the worker session id so the plugin can point the
        # TUI's native task card at the active worker
        dispatches = [l for l in lines if l["type"] == "dispatch"]
        self.assertTrue(all(l.get("session", "").startswith("ses_") for l in dispatches))

    # -- stop/resume + hygiene ------------------------------------------------------------

    def test_request_stop_pauses_job(self):
        self.server.responses = {
            "agent-code": [reply("DONE")],
            "agent-test": [reply("DONE")],
        }
        lm, job_id = self.make_loom()
        lm.stop_flag.set()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "paused")

    def test_ledger_upsert_task_reuses_row(self):
        _, job_id = self.make_loom()
        a = self.led.upsert_task(job_id, "agent-code", "implement", session_id="s1")
        b = self.led.upsert_task(job_id, "agent-code", "implement", last_status="DONE")
        self.assertEqual(a, b)
        row = self.led.task(job_id, "agent-code", "implement")
        self.assertEqual(row["session_id"], "s1")
        self.assertEqual(row["last_status"], "DONE")

    def test_clean_deletes_old_jobs_and_their_sessions(self):
        self.server.responses = {
            "agent-code": [reply("DONE")],
            "agent-test": [reply("DONE")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        # age the job out and clean
        self.led.db.execute("UPDATE jobs SET updated=0 WHERE id=?", (job_id,))
        self.led.db.commit()

        class Args:
            db = self.dbfile
            older_days = 1

        rc = loom.cmd_clean(Args())
        self.assertEqual(rc, 0)
        self.assertIsNone(self.led.job(job_id))
        self.assertEqual(len(self.server.deleted), 3)

    def test_model_error_fails_loudly_not_as_malformed(self):
        # A provider/model failure (bad ref, auth, quota) must surface as a job
        # ERROR with the API message -- never burn nudges as "malformed reply".
        # (Found live: a deprecated model ref walked the whole escalation ladder.)
        self.server.responses = {"agent-code": [MODEL_ERROR]}
        lm, job_id = self.make_loom()
        with self.assertRaises(loom.LoomServerError) as ctx:
            lm.run()
        self.assertIn("model gone", str(ctx.exception))
        job = self.led.job(job_id)
        self.assertEqual(job["status"], "error")
        self.assertIn("model gone", job["report"])
        # exactly one prompt was sent: no nudge, no escalation ladder
        self.assertEqual(len(self.server.prompts("agent-code")), 1)

    def test_ask_routes_one_question_to_research(self):
        # Questions are not features: `ask` dispatches exactly one agent-research
        # child (which has web access) and reports its cited answer -- the loom
        # agent never answers from its own weights.
        self.server.responses = {
            "agent-research": [reply("DONE", "Bananas: rich in potassium [usda.gov]",
                                     "usda.gov/banana, healthline.com/banana")],
        }

        class Args:
            attach = self.server.url
            parent = "ses_parent"
            question = "current nutrition facts for bananas?"
            dir = self.tmp.name
            worker_model = None
            json_events = True
            db = self.dbfile

        rc = loom.cmd_ask(Args())
        self.assertEqual(rc, 0)
        prompts = self.server.prompts()
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0][1], "agent-research")
        self.assertIn("nutrition facts", prompts[0][2])
        led = loom.Ledger(self.dbfile)
        job = led.jobs(limit=1)[0]
        led.close()
        self.assertEqual(job["status"], "done")
        self.assertIn("usda.gov", job["report"])
        # the research child is parented under the loom session, like all workers
        self.assertEqual(self.server.created()[0]["parentID"], "ses_parent")

    def test_template_echo_counts_as_malformed_and_gets_nudged(self):
        # Seen live on the DGX: a weak worker copied the contract's placeholder
        # text verbatim ("RESULT: what you did or found, ..."). That must parse as
        # malformed and trigger the nudge -- never ship as the report.
        echo = ("RESULT: what you did or found, with files as path:line\n"
                "EVIDENCE: each command or check you ran and what it RETURNED, or your sources\n"
                "STATUS: DONE")
        self.assertIsNone(loom.parse_contract(echo)["status"])
        self.server.responses = {
            "agent-research": [echo, reply("DONE", "skill file has 128 lines", "read SKILL.md")],
        }

        class Args:
            attach = self.server.url
            parent = "ses_parent"
            question = "what is in the skill file?"
            dir = self.tmp.name
            worker_model = None
            json_events = True
            db = self.dbfile

        rc = loom.cmd_ask(Args())
        self.assertEqual(rc, 0)
        prompts = self.server.prompts("agent-research")
        self.assertEqual(len(prompts), 2)
        self.assertIn("never the placeholder text", prompts[1][2])
        led = loom.Ledger(self.dbfile)
        job = led.jobs(limit=1)[0]
        led.close()
        self.assertIn("128 lines", job["report"])
        self.assertNotIn("what you did or found", job["report"])

    def test_server_gone_pauses_resumably_instead_of_error(self):
        # A TUI restart (dead port) mid-job is routine -- pause with an --attach
        # hint, never a terminal error.
        tp = loom.Transport("http://127.0.0.1:9", retries=1)  # port 9: refused
        job_id = self.led.create_job(dir=self.tmp.name, server_url="http://127.0.0.1:9",
                                     parent_session="ses_parent", goal="g", risk="simple")
        lm = loom.Loom(self.led, tp, job_id, cfg=self.cfg, out=open(os.devnull, "w"))
        lm.run()
        job = self.led.job(job_id)
        self.assertEqual(job["status"], "paused")
        self.assertIn("--attach", job["report"])

    def test_run_returns_nonzero_when_not_done(self):
        blocked = reply("BLOCKED", "stuck", "no")
        self.server.responses = {"agent-code": [blocked] * 5,
                                 "agent-architect": [blocked] * 5}

        class Args:
            attach = self.server.url
            parent = "ses_parent"
            risk = "simple"
            packet = "goal text"
            dir = self.tmp.name
            worker_model = None
            json_events = True
            db = self.dbfile

        rc = loom.cmd_run(Args())
        self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
