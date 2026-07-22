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


def reply(status, result="ok", evidence="checks ran", next_steps="", extra=""):
    """A well-formed Return-Contract worker reply."""
    lines = [f"RESULT: {result}", f"EVIDENCE: {evidence}", extra,
             f"STATUS: {status}"]
    if next_steps:
        lines.append(f"NEXT STEPS FOR team: {next_steps}")
    return "\n".join(l for l in lines if l)


class FakeOpenCode:
    """Scripted server: responses[agent] is a FIFO of reply texts."""

    def __init__(self):
        self.responses = {}
        self.requests = []          # (method, path, body) in arrival order
        self.sessions = []          # created session dicts
        self.aborted = []
        self.deleted = []
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
        p = loom.parse_contract(reply("DONE", "changed a.py:12", "pytest: 3 passed",
                                      "Send to agent-test."))
        self.assertEqual(p["status"], "DONE")
        self.assertIn("a.py:12", p["result"])
        self.assertIn("3 passed", p["evidence"])
        self.assertEqual(p["next_steps"], "Send to agent-test.")

    def test_parse_contract_tolerates_backticks_and_case(self):
        p = loom.parse_contract("stuff\n`STATUS`: `blocked`\nNEXT STEPS FOR team: x")
        self.assertEqual(p["status"], "BLOCKED")

    def test_parse_contract_malformed(self):
        self.assertIsNone(loom.parse_contract("I did some things, all good!")["status"])

    def test_parse_research_requests(self):
        text = ("RESULT: need facts\nEVIDENCE: none\nSTATUS: NEEDS_RESEARCH\n"
                "RESEARCH REQUEST:\n1. What DB engine is used?\n2. Which auth lib?\n\n"
                "NEXT STEPS FOR team: Send these questions to agent-research")
        p = loom.parse_contract(text)
        self.assertEqual(p["research"],
                         ["What DB engine is used?", "Which auth lib?"])

    def test_parse_findings_and_clean(self):
        text = ("RESULT:\nFINDING 1: a.py:3 missing null check | PASS CONDITION: check added\n"
                "FINDING 2: b.py:9 secret logged | PASS CONDITION: removed\n"
                "EVIDENCE: read diff\nSTATUS: DONE\nNEXT STEPS FOR team: Fix these ONE AT A TIME.")
        p = loom.parse_contract(text)
        self.assertEqual([n for n, _ in p["findings"]], [1, 2])
        self.assertIn("null check", p["findings"][0][1])
        self.assertTrue(loom.parse_contract("No blocking findings.\nSTATUS: DONE")["clean"])

    # -- happy path -----------------------------------------------------------------

    def test_simple_happy_path(self):
        self.server.responses = {
            "agent-code": [reply("DONE", "implemented x", next_steps="Send to agent-test.")],
            "agent-test": [reply("DONE", "all pass", "34/34 PASS",
                                 next_steps="Send to agent-review.")],
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
            "agent-test": [reply("DONE", next_steps="Send to agent-review.")],
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

    # -- failure loops ---------------------------------------------------------------

    def test_test_failure_routes_back_to_same_code_session(self):
        self.server.responses = {
            "agent-code": [reply("DONE", "implemented"),
                           reply("DONE", "fixed the assert")],
            "agent-test": [reply("DONE", "1 failure", "FAIL test_x: boom",
                                 next_steps="Send the failures above to the same agent-code session that made this change."),
                           reply("DONE", "all pass", "34/34 PASS",
                                 next_steps="Send to agent-review.")],
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
                         "STATUS: DONE\nNEXT STEPS FOR team: Fix these ONE AT A TIME.")
        second = ("FINDING 2: b.py:9 secret log | PASS CONDITION: removed\n"
                  "STATUS: DONE\nNEXT STEPS FOR team: Fix these ONE AT A TIME.")
        self.server.responses = {
            "agent-code": [reply("DONE", "implemented"),
                           reply("DONE", "fixed finding 1"),
                           reply("DONE", "fixed finding 2")],
            "agent-test": [reply("DONE", next_steps="Send to agent-review."),
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
            "agent-test": [reply("DONE", next_steps="Send to agent-review.")],
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
            "agent-test": [reply("DONE", next_steps="Send to agent-review.")],
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
            "agent-test": [reply("DONE", next_steps="Send to agent-review.")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom()
        lm.run()
        self.assertEqual(self.led.job(job_id)["status"], "done")
        code = self.server.prompts("agent-code")
        self.assertEqual(len(code), 2)
        self.assertEqual(code[0][0], code[1][0])
        self.assertIn("missing the Return Contract", code[1][2])

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
            "agent-test": [reply("DONE", next_steps="Send to agent-review.")],
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
            "agent-test": [reply("DONE", next_steps="Send to agent-review.")],
            "agent-review": ["No blocking findings.\nSTATUS: DONE"],
        }
        lm, job_id = self.make_loom(worker_model="google/gemini-cheap")
        lm.run()
        for _, agent, _, model in self.server.prompts():
            self.assertEqual(model, {"providerID": "google", "modelID": "gemini-cheap"},
                             f"{agent} must run on the override model")

    # -- stop/resume + hygiene ------------------------------------------------------------

    def test_request_stop_pauses_job(self):
        self.server.responses = {
            "agent-code": [reply("DONE")],
            "agent-test": [reply("DONE", next_steps="Send to agent-review.")],
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
            "agent-test": [reply("DONE", next_steps="Send to agent-review.")],
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
