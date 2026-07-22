"""omw_loom.py -- the loom: a deterministic pipeline conductor for the omw agent roster.

The loom replaces the v1 `team` LLM-orchestrator's CONTROL FLOW with code, while
keeping every worker (agent-research/-code/-test/-instruct/-architect/-review)
exactly as `omw sync` configures them. Workers keep their skills and the Return
Contract; the loom is simply a new consumer of that contract:

    RESULT: ...            <- opaque text, couriered between sessions
    EVIDENCE: ...          <- opaque text + the anti-spin comparison key
    STATUS: DONE | CONTINUE | NEEDS_RESEARCH | BLOCKED     <- parsed, routed on
    NEXT STEPS FOR team: ...                               <- advisory, logged

Workers run as REAL OpenCode sessions created over the server HTTP API as
CHILDREN of the loom agent's session (parentID), so the TUI renders them with
its native subagent navigation (right/left arrows, "Subagent i/N" footer).

Pipeline (fixed graph; judgment stays in the LLM workers):

    intake (loom agent) -> [plan: agent-architect]* -> code: agent-code
        -> test: agent-test (loop failures back to the same code session)
        -> review: agent-review (fix findings ONE AT A TIME, new code session
           per finding, same review session for re-review) -> done
    (* plan phase only when risk != simple; NEEDS_RESEARCH fans out parallel
       agent-research sessions and resumes the requesting session.)

Escalation ladder, per stuck step (config `loom.max_attempts` caps the climb):
    1. retry in the same session with the exact failure evidence
    2. fresh session, same model, distilled packet (context-poison reset)
    3. same step on the next model up the default_models.json ranking
    4. pause the job for the human (`omw loom resume --job N` to continue)

State lives in SQLite (WAL) so a killed/aborted job resumes exactly where it
stopped; sessions persist server-side so resumed workers keep their context.

Stdlib only. Loaded by path from omodel-wire.py (like utils/omw_proxy.py).
"""

import json
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Defaults (overridable via wire.json {"loom": {...}} -- see loom_config()).
# ---------------------------------------------------------------------------
DEFAULTS = {
    "step_timeout": 1800,       # seconds one worker step may run before abort
    "max_attempts": 3,          # escalation-ladder rungs before pausing (>=4 never pauses early)
    "max_test_cycles": 3,       # test-fail -> fix -> retest loops before escalating
    "max_fix_findings": 10,     # review findings fixed before pausing (runaway guard)
    "research_parallel": 4,     # concurrent agent-research sessions in a fan-out
    "nudge_malformed": True,    # one reprompt when a reply is missing STATUS:
}

WORKERS = ("agent-research", "agent-code", "agent-test",
           "agent-instruct", "agent-architect", "agent-review")

STATUSES = ("DONE", "CONTINUE", "NEEDS_RESEARCH", "BLOCKED")


def data_dir():
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "otools")
    os.makedirs(d, exist_ok=True)
    return d


def db_path():
    return os.environ.get("OMW_LOOM_DB") or os.path.join(data_dir(), "loom.db")


def loom_config(wire_settings=None):
    """DEFAULTS overlaid with wire.json's {"loom": {...}} section, if present."""
    cfg = dict(DEFAULTS)
    if isinstance(wire_settings, dict):
        for k, v in (wire_settings.get("loom") or {}).items():
            if k in cfg:
                cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Return Contract parsing. Tolerant: workers are cheap models; the parser is
# the reliable half of the conversation.
# ---------------------------------------------------------------------------
_STATUS_RE = re.compile(r"^\s*`?STATUS`?\s*:\s*`?(DONE|CONTINUE|NEEDS_RESEARCH|BLOCKED)`?\s*$",
                        re.MULTILINE | re.IGNORECASE)
_NEXT_RE = re.compile(r"^\s*`?NEXT STEPS FOR team`?\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*(RESULT|EVIDENCE)\s*:\s*", re.MULTILINE)
_RESEARCH_RE = re.compile(r"RESEARCH REQUEST\s*:?\s*(.*)", re.IGNORECASE)
_FINDING_RE = re.compile(r"^\s*FINDING\s+(\d+)\s*:\s*(.+?)(?=^\s*FINDING\s+\d+\s*:|\Z)",
                         re.MULTILINE | re.DOTALL)
_CLEAN_RE = re.compile(r"No blocking findings|Review passed", re.IGNORECASE)


def parse_contract(text):
    """-> dict(status, next_steps, result, evidence, research, findings, clean).
    status is None when the reply is malformed (no STATUS line)."""
    text = text or ""
    m = _STATUS_RE.search(text)
    status = m.group(1).upper() if m else None
    nm = _NEXT_RE.search(text)
    next_steps = nm.group(1).strip() if nm else ""

    def section(name):
        sm = re.search(r"^\s*%s\s*:\s*(.*?)(?=^\s*(?:RESULT|EVIDENCE|STATUS|NEXT STEPS FOR team)\s*:|\Z)"
                       % name, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        return sm.group(1).strip() if sm else ""

    research = []
    rm = _RESEARCH_RE.search(text)
    if rm:
        # Questions are the numbered/bulleted lines following RESEARCH REQUEST:
        tail = text[rm.start():]
        inline = rm.group(1).strip()
        if inline and not inline.startswith(("-", "*", "1")):
            research.append(inline)
        for line in tail.splitlines()[1:]:
            ls = line.strip()
            if re.match(r"^(?:[-*]|\d+[.)])\s+\S", ls):
                research.append(re.sub(r"^(?:[-*]|\d+[.)])\s+", "", ls))
            elif research and not ls:
                break
    findings = [(int(n), body.strip()) for n, body in _FINDING_RE.findall(text)]
    return {
        "status": status,
        "next_steps": next_steps,
        "result": section("RESULT"),
        "evidence": section("EVIDENCE"),
        "research": research,
        "findings": findings,
        "clean": bool(_CLEAN_RE.search(text)),
    }


def evidence_key(parsed):
    """Stable key for the anti-spin comparison: same EVIDENCE twice = no progress."""
    return re.sub(r"\s+", " ", (parsed.get("evidence") or parsed.get("result") or "")).strip().lower()


# ---------------------------------------------------------------------------
# OpenCode server transport (urllib; retries connection errors, not HTTP ones).
# ---------------------------------------------------------------------------
class Transport:
    def __init__(self, base_url, timeout=30, retries=3):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _call(self, method, path, body=None, timeout=None):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=data, method=method,
                                         headers={"content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    raw = r.read().decode("utf-8", "replace")
                    return json.loads(raw) if raw.strip() else None
            except urllib.error.HTTPError as e:
                # HTTP errors are real answers -- do not retry blindly.
                detail = e.read().decode("utf-8", "replace")[:400]
                raise LoomServerError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
                time.sleep(min(2 ** attempt, 5))
        raise LoomServerError(f"{method} {path} unreachable after {self.retries} tries: {last}")

    def sessions(self):
        return self._call("GET", "/session") or []

    def create_session(self, parent_id, title, directory):
        q = "?" + urllib.parse.urlencode({"directory": directory}) if directory else ""
        return self._call("POST", "/session" + q,
                          {"parentID": parent_id, "title": title})

    def prompt(self, session_id, text, agent, model=None, timeout=None):
        body = {"agent": agent, "parts": [{"type": "text", "text": text}]}
        if model:
            prov, _, mid = model.partition("/")
            body["model"] = {"providerID": prov, "modelID": mid}
        resp = self._call("POST", f"/session/{session_id}/message", body, timeout=timeout)
        parts = (resp or {}).get("parts") or []
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
        err = ((resp or {}).get("info") or {}).get("error")
        if err and not text:
            # A model/provider-level failure (bad model ref, auth, quota) is a
            # configuration problem, not a malformed worker reply -- surface it.
            detail = err.get("data", {}).get("message") or err.get("name") or str(err)
            raise LoomServerError(f"model error on {agent}"
                                  f"{' (' + model + ')' if model else ''}: {detail[:300]}")
        return text

    def abort(self, session_id):
        try:
            self._call("POST", f"/session/{session_id}/abort", {})
        except LoomServerError:
            pass  # best-effort: aborting an idle session is fine to fail

    def delete_session(self, session_id):
        self._call("DELETE", f"/session/{session_id}")


class LoomServerError(RuntimeError):
    pass


class LoomPaused(Exception):
    """Raised inside the state machine to stop cleanly with the job left resumable."""


# ---------------------------------------------------------------------------
# Ledger (SQLite, WAL). One file, five tables; every transition is a row.
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, created REAL, updated REAL,
  dir TEXT, server_url TEXT, parent_session TEXT,
  goal TEXT, risk TEXT, phase TEXT, status TEXT,
  plan TEXT, worker_model TEXT, report TEXT);
CREATE TABLE IF NOT EXISTS tasks(
  id INTEGER PRIMARY KEY, job_id INTEGER, role TEXT, purpose TEXT,
  session_id TEXT, attempts INTEGER DEFAULT 0, model TEXT,
  last_status TEXT, last_evidence_key TEXT, spin INTEGER DEFAULT 0,
  created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS findings(
  id INTEGER PRIMARY KEY, job_id INTEGER, n INTEGER,
  text TEXT, status TEXT DEFAULT 'pending');
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY, job_id INTEGER, ts REAL, text TEXT,
  consumed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, job_id INTEGER, ts REAL, kind TEXT, detail TEXT);
"""


class Ledger:
    """Thread-safe: the research fan-out writes from worker threads, so the
    connection allows cross-thread use and every method serializes on a lock."""

    def __init__(self, path=None):
        self.path = path or db_path()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
            self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    def create_job(self, **kw):
        now = time.time()
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO jobs(created,updated,dir,server_url,parent_session,goal,risk,"
                "phase,status,worker_model) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (now, now, kw.get("dir"), kw.get("server_url"), kw.get("parent_session"),
                 kw.get("goal"), kw.get("risk", "medium"), "intake", "running",
                 kw.get("worker_model")))
            self.db.commit()
            return cur.lastrowid

    def job(self, job_id):
        with self.lock:
            return self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def jobs(self, limit=20):
        with self.lock:
            return self.db.execute(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def update_job(self, job_id, **kw):
        kw["updated"] = time.time()
        sets = ",".join(f"{k}=?" for k in kw)
        with self.lock:
            self.db.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*kw.values(), job_id))
            self.db.commit()

    def upsert_task(self, job_id, role, purpose, **kw):
        now = time.time()
        with self.lock:
            row = self.db.execute(
                "SELECT id FROM tasks WHERE job_id=? AND role=? AND purpose=?",
                (job_id, role, purpose)).fetchone()
            if row:
                kw["updated"] = now
                sets = ",".join(f"{k}=?" for k in kw)
                self.db.execute(f"UPDATE tasks SET {sets} WHERE id=?", (*kw.values(), row["id"]))
                self.db.commit()
                return row["id"]
            cur = self.db.execute(
                "INSERT INTO tasks(job_id,role,purpose,session_id,attempts,model,last_status,"
                "last_evidence_key,spin,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, role, purpose, kw.get("session_id"), kw.get("attempts", 0),
                 kw.get("model"), kw.get("last_status"), kw.get("last_evidence_key"),
                 kw.get("spin", 0), now, now))
            self.db.commit()
            return cur.lastrowid

    def task(self, job_id, role, purpose):
        with self.lock:
            return self.db.execute(
                "SELECT * FROM tasks WHERE job_id=? AND role=? AND purpose=?",
                (job_id, role, purpose)).fetchone()

    def tasks(self, job_id):
        with self.lock:
            return self.db.execute(
                "SELECT * FROM tasks WHERE job_id=? ORDER BY id", (job_id,)).fetchall()

    def add_findings(self, job_id, findings):
        with self.lock:
            self.db.execute("DELETE FROM findings WHERE job_id=? AND status='pending'", (job_id,))
            for n, txt in findings:
                self.db.execute("INSERT INTO findings(job_id,n,text) VALUES(?,?,?)",
                                (job_id, n, txt))
            self.db.commit()

    def pending_findings(self, job_id):
        with self.lock:
            return self.db.execute(
                "SELECT * FROM findings WHERE job_id=? AND status='pending' ORDER BY n",
                (job_id,)).fetchall()

    def finding_done(self, fid):
        with self.lock:
            self.db.execute("UPDATE findings SET status='fixed' WHERE id=?", (fid,))
            self.db.commit()

    def add_note(self, job_id, text):
        with self.lock:
            self.db.execute("INSERT INTO notes(job_id,ts,text) VALUES(?,?,?)",
                            (job_id, time.time(), text))
            self.db.commit()

    def drain_notes(self, job_id):
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM notes WHERE job_id=? AND consumed=0 ORDER BY id",
                (job_id,)).fetchall()
            if rows:
                self.db.execute("UPDATE notes SET consumed=1 WHERE job_id=? AND consumed=0",
                                (job_id,))
                self.db.commit()
            return [r["text"] for r in rows]

    def event(self, job_id, kind, detail=""):
        with self.lock:
            self.db.execute("INSERT INTO events(job_id,ts,kind,detail) VALUES(?,?,?,?)",
                            (job_id, time.time(), kind, detail))
            self.db.commit()

    def events(self, job_id, after_id=0):
        with self.lock:
            return self.db.execute(
                "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id",
                (job_id, after_id)).fetchall()


# ---------------------------------------------------------------------------
# The conductor.
# ---------------------------------------------------------------------------
class Loom:
    """One job's run. Emits events to the ledger and (optionally) JSON lines on
    stdout for the OpenCode plugin tool to relay into the TUI card."""

    def __init__(self, ledger, transport, job_id, cfg=None, model_ranking=None,
                 json_events=False, out=None):
        self.led = ledger
        self.tp = transport
        self.job_id = job_id
        self.cfg = cfg or dict(DEFAULTS)
        # model_ranking: {role_agent_name: [model_ref, ...]} from default_models.json
        self.ranking = model_ranking or {}
        self.json_events = json_events
        self.out = out or sys.stdout
        self.stop_flag = threading.Event()
        self.active_session = None

    # ---- event/progress plumbing -------------------------------------------------
    def emit(self, kind, detail="", title=None):
        self.led.event(self.job_id, kind, detail)
        if self.json_events:
            print(json.dumps({"type": kind, "detail": detail, "title": title or detail}),
                  file=self.out, flush=True)
        else:
            print(f"[loom #{self.job_id}] {kind}: {detail}", file=self.out, flush=True)

    def request_stop(self):
        """Signal handler / abort path: finish the in-flight call, then pause."""
        self.stop_flag.set()
        if self.active_session:
            self.tp.abort(self.active_session)

    def _check_stop(self):
        if self.stop_flag.is_set():
            raise LoomPaused("stop requested")

    # ---- worker dispatch ----------------------------------------------------------
    def _model_for(self, role, bump=0):
        """Job-wide override wins; else ranking[role][bump]; else agent default (None)."""
        job = self.led.job(self.job_id)
        if job["worker_model"]:
            return job["worker_model"]
        ranked = self.ranking.get(role) or []
        if bump > 0 and len(ranked) > bump:
            return ranked[bump]
        return None  # let the server use the agent's configured model

    def _notes_suffix(self):
        notes = self.led.drain_notes(self.job_id)
        if not notes:
            return ""
        return "\n\nOPERATOR NOTES (from the human, incorporate them):\n" + \
               "\n".join(f"- {n}" for n in notes)

    def _dispatch(self, role, purpose, prompt, resume=False, fresh=False, bump=0):
        """Send one prompt to a worker session; create/resume per flags.
        Returns (parsed_contract, raw_text). Handles the malformed-nudge."""
        self._check_stop()
        job = self.led.job(self.job_id)
        task = self.led.task(self.job_id, role, purpose)
        model = self._model_for(role, bump=bump)

        if task and task["session_id"] and resume and not fresh:
            sid = task["session_id"]
        else:
            title = f"{purpose} (@{role} subagent)"
            sess = self.tp.create_session(job["parent_session"], title, job["dir"])
            sid = sess["id"]
            self.led.upsert_task(self.job_id, role, purpose, session_id=sid, model=model)
            self.emit("session", f"{role} [{purpose}] -> {sid}" + (f" on {model}" if model else ""))

        prompt = prompt + self._notes_suffix()
        self.emit("dispatch", f"{role} [{purpose}]" + (" (resume)" if resume and not fresh else ""),
                  title=f"{purpose}: {role} working...")
        self.active_session = sid
        try:
            text = self._timed_prompt(sid, prompt, role, model)
        finally:
            self.active_session = None
        parsed = parse_contract(text)

        if parsed["status"] is None and self.cfg.get("nudge_malformed", True):
            self.emit("malformed", f"{role} reply missing STATUS; nudging once")
            self.active_session = sid
            try:
                text = self._timed_prompt(
                    sid, "Your reply was missing the Return Contract. Repeat your findings and "
                         "end with the RESULT / EVIDENCE / STATUS / NEXT STEPS FOR team lines "
                         "exactly as your role skill specifies.", role, model)
            finally:
                self.active_session = None
            parsed = parse_contract(text)
        if parsed["status"] is None:
            parsed["status"] = "BLOCKED"
            parsed["evidence"] = parsed["evidence"] or "worker returned malformed output twice"
            self.emit("blocked", f"{role} [{purpose}]: malformed output twice -> BLOCKED")

        self.led.upsert_task(self.job_id, role, purpose, last_status=parsed["status"])
        self.emit("status", f"{role} [{purpose}] -> {parsed['status']}"
                            + (f" | next: {parsed['next_steps']}" if parsed["next_steps"] else ""),
                  title=f"{purpose}: {role} -> {parsed['status']}")
        return parsed, text

    def _timed_prompt(self, sid, prompt, role, model):
        """Run the blocking prompt POST in a thread so stop/timeout can abort it."""
        result, error = {}, {}

        def work():
            try:
                result["text"] = self.tp.prompt(sid, prompt, role, model=model,
                                                timeout=self.cfg["step_timeout"] + 60)
            except Exception as e:  # noqa: BLE001 -- reported to the state machine
                error["e"] = e

        t = threading.Thread(target=work, daemon=True)
        t.start()
        deadline = time.time() + self.cfg["step_timeout"]
        while t.is_alive():
            t.join(timeout=1.0)
            if self.stop_flag.is_set() or time.time() > deadline:
                self.tp.abort(sid)
                t.join(timeout=30)
                if self.stop_flag.is_set():
                    raise LoomPaused("stopped during worker step")
                self.emit("timeout", f"{role} step exceeded {self.cfg['step_timeout']}s; aborted")
                return result.get("text", "") or ""
        if "e" in error:
            raise error["e"]
        return result.get("text", "")

    # ---- escalation ladder ----------------------------------------------------------
    def _escalate(self, role, purpose, packet, why):
        """Climb one rung per call. attempts: 1=same-session retry happened upstream,
        2=fresh session, 3=model bump, 4+=pause for the human."""
        task = self.led.task(self.job_id, role, purpose)
        attempts = (task["attempts"] if task else 0) + 1
        self.led.upsert_task(self.job_id, role, purpose, attempts=attempts)
        if attempts >= self.cfg["max_attempts"] + 1:
            self.led.update_job(self.job_id, status="paused",
                                report=f"paused: {role} [{purpose}] stuck after "
                                       f"{attempts - 1} attempts ({why}). "
                                       f"Resume with: omw loom resume --job {self.job_id} "
                                       f"--note '<your guidance>'")
            self.emit("paused", f"{role} [{purpose}] stuck ({why}). "
                                f"Resume with: omw loom resume --job {self.job_id}",
                      title=f"paused: {role} needs a human")
            raise LoomPaused(why)
        if attempts <= 1:
            self.emit("escalate", f"{role} [{purpose}]: fresh session, distilled packet ({why})")
            return self._dispatch(role, purpose, packet, fresh=True)
        # attempts >= 2 -> model bump (index into the role's ranking; falls back to a
        # fresh default-model session when the ranking has no higher rung)
        self.emit("escalate", f"{role} [{purpose}]: retry on next-ranked model ({why})")
        return self._dispatch(role, purpose, packet, fresh=True, bump=attempts - 1)

    # ---- research fan-out -------------------------------------------------------------
    def _research(self, questions):
        """Parallel agent-research children; returns combined findings text.
        Thread errors are collected and re-raised (never swallowed)."""
        results = [None] * len(questions)
        errors = []
        done = [0]
        lock = threading.Lock()
        sem = threading.Semaphore(self.cfg["research_parallel"])

        def one(i, q):
            with sem:
                if self.stop_flag.is_set():
                    return
                try:
                    parsed, raw = self._dispatch(
                        "agent-research", f"research {self.job_id}.{i + 1}",
                        f"Answer this question for the current repository/goal.\n\n"
                        f"QUESTION: {q}")
                    results[i] = f"QUESTION: {q}\nANSWER:\n{parsed['result'] or raw}"
                    with lock:
                        done[0] += 1
                        n = done[0]
                    # Keep the live tool card honest about parallel progress.
                    self.emit("research", f"answered {n}/{len(questions)}",
                              title=f"research: {n}/{len(questions)} answered "
                                    f"({len(questions) - n} running)")
                except LoomPaused:
                    pass
                except Exception as e:  # noqa: BLE001 -- re-raised below
                    errors.append(e)

        threads = [threading.Thread(target=one, args=(i, q), daemon=True)
                   for i, q in enumerate(questions)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._check_stop()
        if errors and not any(results):
            raise errors[0]
        return "\n\n".join(r for r in results if r)

    # ---- the state machine ---------------------------------------------------------------
    def run_ask(self, question):
        """Single research dispatch -- the loom's Q&A path (no pipeline). Routes a
        factual/current-information question to one agent-research child so the
        loom agent never answers from its own weights."""
        try:
            parsed, raw = self._dispatch(
                "agent-research", "ask",
                "Answer this question with current, verified information; cite your "
                f"sources.\n\nQUESTION: {question}")
            report = ("LOOM RESEARCH ANSWER\n\n"
                      f"{parsed['result'] or raw}\n\n"
                      f"EVIDENCE/SOURCES:\n{parsed['evidence'] or '(inline above)'}")
            self.led.update_job(self.job_id, status="done", phase="done", report=report)
            self.emit("done", "research answered", title="answered")
            if self.json_events:
                print(json.dumps({"type": "report", "report": report}),
                      file=self.out, flush=True)
        except LoomPaused:
            self.led.update_job(self.job_id, status="paused", report="paused by operator")
            self.emit("paused", f"ask job {self.job_id} paused")
        except LoomServerError as e:
            self.led.update_job(self.job_id, status="error", report=str(e))
            self.emit("error", str(e))
            raise

    def run(self):
        try:
            self._run()
        except LoomPaused:
            job = self.led.job(self.job_id)
            if job["status"] == "running":
                self.led.update_job(self.job_id, status="paused",
                                    report="paused by operator")
            self.emit("paused", f"job {self.job_id} paused; resume with: "
                                f"omw loom resume --job {self.job_id}")
        except LoomServerError as e:
            self.led.update_job(self.job_id, status="error", report=str(e))
            self.emit("error", str(e))
            raise

    def _run(self):
        job = self.led.job(self.job_id)
        phase = job["phase"]
        if phase in ("intake", "plan"):
            self._phase_plan()
            self.led.update_job(self.job_id, phase="code")
        if self.led.job(self.job_id)["phase"] == "code":
            self._phase_code()
            self.led.update_job(self.job_id, phase="test")
        if self.led.job(self.job_id)["phase"] == "test":
            self._phase_test()
            self.led.update_job(self.job_id, phase="review")
        if self.led.job(self.job_id)["phase"] == "review":
            self._phase_review()
        self._finish()

    def _packet(self, extra=""):
        job = self.led.job(self.job_id)
        parts = [f"GOAL: {job['goal']}"]
        if job["plan"]:
            parts.append(f"PLAN AND ACCEPTANCE CRITERIA (from agent-architect):\n{job['plan']}")
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)

    def _phase_plan(self):
        job = self.led.job(self.job_id)
        if job["risk"] == "simple":
            self.emit("phase", "plan skipped (risk=simple)")
            return
        self.led.update_job(self.job_id, phase="plan")
        self.emit("phase", "plan: agent-architect", title="planning...")
        prompt = (f"Plan this work.\n\n{self._packet()}\n\n"
                  "Return your numbered plan, checkable acceptance criteria, scope "
                  "exclusions, and verification commands.")
        parsed, raw = self._dispatch("agent-architect", "plan", prompt)
        rounds = 0
        while parsed["status"] == "NEEDS_RESEARCH" and rounds < 3:
            self._check_stop()
            rounds += 1
            qs = parsed["research"] or [parsed["next_steps"] or "the facts you listed"]
            self.emit("research", f"fan-out: {len(qs)} question(s)")
            answers = self._research(qs)
            parsed, raw = self._dispatch(
                "agent-architect", "plan",
                f"Research results:\n\n{answers}\n\nNow finish the plan.", resume=True)
        if parsed["status"] != "DONE":
            parsed, raw = self._escalate("agent-architect", "plan",
                                         f"Plan this work.\n\n{self._packet()}",
                                         f"architect returned {parsed['status']}")
            if parsed["status"] != "DONE":
                self._escalate("agent-architect", "plan", self._packet(),
                               f"architect returned {parsed['status']} again")
        self.led.update_job(self.job_id, plan=parsed["result"] or raw)
        self.emit("plan", "plan captured")

    def _code_step(self, purpose, prompt, resume=False):
        """agent-code with the CONTINUE / NEEDS_RESEARCH / BLOCKED loop handled."""
        parsed, raw = self._dispatch("agent-code", purpose, prompt, resume=resume)
        while True:
            self._check_stop()
            if parsed["status"] == "DONE":
                return parsed, raw
            if parsed["status"] == "CONTINUE":
                task = self.led.task(self.job_id, "agent-code", purpose)
                key = evidence_key(parsed)
                spin = (task["spin"] or 0) + 1 if task and task["last_evidence_key"] == key else 0
                self.led.upsert_task(self.job_id, "agent-code", purpose,
                                     last_evidence_key=key, spin=spin)
                if spin >= 1:
                    self.emit("antispin", f"agent-code [{purpose}]: CONTINUE with stale "
                                          "EVIDENCE -> treating as BLOCKED")
                    parsed = dict(parsed, status="BLOCKED")
                    continue
                parsed, raw = self._dispatch("agent-code", purpose,
                                             "Continue with your plan.", resume=True)
            elif parsed["status"] == "NEEDS_RESEARCH":
                qs = parsed["research"] or [parsed["next_steps"] or "your stated question"]
                answers = self._research(qs)
                parsed, raw = self._dispatch("agent-code", purpose,
                                             f"Research results:\n\n{answers}\n\nContinue.",
                                             resume=True)
            elif parsed["status"] == "BLOCKED":
                diag_prompt = (f"agent-code is BLOCKED on this feature.\n\n{self._packet()}\n\n"
                               f"BLOCKED REPORT:\nRESULT: {parsed['result']}\n"
                               f"EVIDENCE: {parsed['evidence']}\n\n"
                               "Diagnose the smallest correction that unblocks the goal.")
                dparsed, draw = self._dispatch("agent-architect", "diagnose", diag_prompt,
                                               resume=bool(self.led.task(self.job_id,
                                                                         "agent-architect",
                                                                         "diagnose")))
                if dparsed["status"] != "DONE":
                    parsed, raw = self._escalate("agent-code", purpose, self._packet(),
                                                 "blocked and architect could not diagnose")
                    continue
                parsed, raw = self._dispatch(
                    "agent-code", purpose,
                    f"agent-architect's correction:\n\n{dparsed['result'] or draw}\n\n"
                    "Apply it and finish the task.", resume=True)
            else:
                parsed, raw = self._escalate("agent-code", purpose, self._packet(),
                                             f"unexpected status {parsed['status']}")

    def _phase_code(self):
        self.emit("phase", "code: agent-code", title="coding...")
        prompt = (f"Implement this.\n\n{self._packet()}\n\n"
                  "Run your focused checks; agent-test runs the broad suite after you.")
        parsed, _ = self._code_step("implement", prompt)
        self.led.upsert_task(self.job_id, "agent-code", "implement",
                             last_status="DONE")
        self.emit("code", "implementation complete")
        self._last_code_result = parsed["result"]

    def _phase_test(self):
        self.emit("phase", "test: agent-test", title="testing...")
        result = getattr(self, "_last_code_result", "") or ""
        base = (f"Verify this implementation.\n\n{self._packet()}\n\n"
                f"IMPLEMENTATION SUMMARY (from agent-code):\n{result}\n\n"
                "Run the broad checks and report exact PASS/FAIL evidence.")
        parsed, raw = self._dispatch("agent-test", "verify", base)
        cycles = 0
        while cycles < self.cfg["max_test_cycles"]:
            self._check_stop()
            if self._test_passed(parsed):
                self.emit("test", "checks pass")
                self._last_test_evidence = parsed["evidence"] or parsed["result"] or raw
                return
            cycles += 1
            failure = parsed["evidence"] or parsed["result"] or raw
            self.emit("testfail", f"cycle {cycles}: routing failure back to agent-code")
            fix_prompt = (f"agent-test reports failures. Fix them.\n\nFAILURES:\n{failure}\n\n"
                          "Pass condition: the reported checks succeed.")
            self._code_step("implement", fix_prompt, resume=True)
            parsed, raw = self._dispatch("agent-test", "verify",
                                         "The fix is in. Re-run the failing checks and report.",
                                         resume=True)
        self._escalate("agent-code", "implement", self._packet(),
                       f"{cycles} test cycles without a pass")
        # escalation replaced the coder; give test one final round before pausing next loop
        parsed, raw = self._dispatch("agent-test", "verify",
                                     "Re-run the checks against the current state.", resume=True)
        if not self._test_passed(parsed):
            self.led.update_job(self.job_id, status="paused",
                                report="paused: tests failing after escalation")
            self.emit("paused", "tests still failing after escalation; human needed")
            raise LoomPaused("tests failing after escalation")
        self._last_test_evidence = parsed["evidence"] or parsed["result"] or raw

    @staticmethod
    def _test_passed(parsed):
        nxt = (parsed["next_steps"] or "").lower()
        if "agent-review" in nxt:
            return True
        if "agent-code" in nxt:
            return False
        # fall back to evidence text
        blob = f"{parsed['result']} {parsed['evidence']}".upper()
        return "FAIL" not in blob and parsed["status"] == "DONE"

    def _phase_review(self):
        self.emit("phase", "review: agent-review", title="reviewing...")
        job = self.led.job(self.job_id)
        packet = (f"Review this completed implementation.\n\n{self._packet()}\n\n"
                  f"TEST EVIDENCE (from agent-test):\n"
                  f"{getattr(self, '_last_test_evidence', '(see agent-test session)')}\n\n"
                  "List every blocker/regression as a numbered line in the exact form "
                  "'FINDING <n>: <path:line> <description> | PASS CONDITION: <condition>'. "
                  "If there are none, say 'No blocking findings.'")
        parsed, raw = self._dispatch("agent-review", "review", packet)
        fixed = 0
        # STRICTLY one finding at a time: fix the first pending finding, re-test,
        # re-review (same review session), and only then look at what remains --
        # exactly the v1 agent-team Fix Loop contract the reviewer skill expects.
        while not (parsed["clean"] or not parsed["findings"]):
            self._check_stop()
            self.led.add_findings(self.job_id, parsed["findings"])
            pend = self.led.pending_findings(self.job_id)
            self.emit("findings", f"{len(pend)} blocker/regression finding(s) pending")
            row = pend[0]
            fixed += 1
            if fixed > self.cfg["max_fix_findings"]:
                self.led.update_job(self.job_id, status="paused",
                                    report="paused: finding-fix cap reached")
                self.emit("paused", "fix cap reached; human review needed")
                raise LoomPaused("fix cap")
            self.emit("fix", f"finding {row['n']}: dispatching a NEW agent-code session",
                      title=f"fixing finding {row['n']}...")
            self._code_step(f"fix finding {row['n']}",
                            f"Fix exactly this review finding -- nothing else.\n\n"
                            f"{self._packet()}\n\nFINDING: {row['text']}")
            self._dispatch("agent-test", "verify",
                           f"A fix for review finding {row['n']} landed. Re-run the "
                           "relevant checks and report.", resume=True)
            self.led.finding_done(row["id"])
            parsed, raw = self._dispatch(
                "agent-review", "review",
                f"Finding {row['n']} was fixed and re-tested. Re-review ONLY that finding "
                "and regressions from its fix. List anything still blocking with the same "
                "numbered FINDING format; otherwise say 'No blocking findings.'", resume=True)
        self.emit("review", "no blocking findings")
        self._final_review = parsed["result"] or raw

    def _finish(self):
        tasks = self.led.tasks(self.job_id)
        lines = [f"- {t['role']} [{t['purpose']}]: {t['last_status'] or '-'} "
                 f"(session {t['session_id']}, attempts {t['attempts']})" for t in tasks]
        report = ("LOOM JOB COMPLETE\n"
                  f"GOAL: {self.led.job(self.job_id)['goal']}\n\n"
                  f"REVIEW: {getattr(self, '_final_review', 'clean')}\n\n"
                  "WORKERS:\n" + "\n".join(lines) +
                  "\n\nPR: not created (run `omw loom pr --job %d` after approval)."
                  % self.job_id)
        self.led.update_job(self.job_id, status="done", phase="done", report=report)
        self.emit("done", "job complete", title="done")
        if self.json_events:
            print(json.dumps({"type": "report", "report": report}), file=self.out, flush=True)


# ---------------------------------------------------------------------------
# CLI wiring (called from omodel-wire.py's `omw loom` subcommand).
# ---------------------------------------------------------------------------
def _load_ranking(default_models_path):
    try:
        with open(default_models_path, encoding="utf-8") as f:
            return (json.load(f).get("agents") or {})
    except (OSError, ValueError):
        return {}


def cmd_run(args, wire_settings=None, default_models_path=None):
    cfg = loom_config(wire_settings)
    led = Ledger(getattr(args, "db", None))
    try:
        goal = args.packet if getattr(args, "packet", None) else sys.stdin.read()
        if not goal.strip():
            print("loom: empty packet (use --packet or pipe it on stdin)", file=sys.stderr)
            return 2
        tp = Transport(args.attach)
        job_id = led.create_job(dir=args.dir or os.getcwd(), server_url=args.attach,
                                parent_session=args.parent, goal=goal.strip(),
                                risk=args.risk,
                                worker_model=getattr(args, "worker_model", None))
        loom = Loom(led, tp, job_id, cfg=cfg,
                    model_ranking=_load_ranking(default_models_path) if default_models_path else {},
                    json_events=getattr(args, "json_events", False))
        _install_signals(loom)
        loom.emit("start", f"job {job_id} risk={args.risk} parent={args.parent}")
        loom.run()
        job = led.job(job_id)
        if not getattr(args, "json_events", False):
            print("\n" + (job["report"] or job["status"]))
        return 0 if job["status"] == "done" else 3
    finally:
        led.close()


def cmd_ask(args, wire_settings=None, default_models_path=None):
    led = Ledger(getattr(args, "db", None))
    try:
        question = args.question if getattr(args, "question", None) else sys.stdin.read()
        if not question.strip():
            print("loom: empty question", file=sys.stderr)
            return 2
        tp = Transport(args.attach)
        job_id = led.create_job(dir=args.dir or os.getcwd(), server_url=args.attach,
                                parent_session=args.parent, goal=question.strip(),
                                risk="ask",
                                worker_model=getattr(args, "worker_model", None))
        loom = Loom(led, tp, job_id, cfg=loom_config(wire_settings),
                    model_ranking=_load_ranking(default_models_path) if default_models_path else {},
                    json_events=getattr(args, "json_events", False))
        _install_signals(loom)
        loom.emit("start", f"ask job {job_id}")
        loom.run_ask(question.strip())
        job = led.job(job_id)
        if not getattr(args, "json_events", False):
            print("\n" + (job["report"] or job["status"]))
        return 0 if job["status"] == "done" else 3
    finally:
        led.close()


def cmd_resume(args, wire_settings=None, default_models_path=None):
    led = Ledger(getattr(args, "db", None))
    try:
        job = led.job(args.job)
        if not job:
            print(f"loom: no job {args.job}", file=sys.stderr)
            return 2
        if getattr(args, "note", None):
            led.add_note(args.job, args.note)
        led.update_job(args.job, status="running")
        tp = Transport(getattr(args, "attach", None) or job["server_url"])
        loom = Loom(led, tp, args.job, cfg=loom_config(wire_settings),
                    model_ranking=_load_ranking(default_models_path) if default_models_path else {},
                    json_events=getattr(args, "json_events", False))
        _install_signals(loom)
        loom.emit("resume", f"job {args.job} at phase {job['phase']}")
        loom.run()
        job = led.job(args.job)
        if not getattr(args, "json_events", False):
            print("\n" + (job["report"] or job["status"]))
        return 0 if job["status"] == "done" else 3
    finally:
        led.close()


def cmd_status(args):
    led = Ledger(getattr(args, "db", None))
    try:
        if getattr(args, "job", None):
            job = led.job(args.job)
            if not job:
                print(f"loom: no job {args.job}", file=sys.stderr)
                return 2
            print(f"job {job['id']}  [{job['status']}]  phase={job['phase']}  risk={job['risk']}")
            print(f"  goal: {job['goal'][:100]}")
            for t in led.tasks(args.job):
                print(f"  {t['role']:16s} [{t['purpose']}] {t['last_status'] or '-':14s} "
                      f"session={t['session_id']} attempts={t['attempts']}")
            if job["report"]:
                print(f"\n{job['report']}")
            return 0
        for job in led.jobs():
            print(f"#{job['id']:<4d} [{job['status']:8s}] {job['phase']:8s} {job['goal'][:70]}")
        return 0
    finally:
        led.close()


def cmd_watch(args):
    led = Ledger(getattr(args, "db", None))
    try:
        job_id = getattr(args, "job", None)
        if not job_id:
            rows = led.jobs(limit=1)
            if not rows:
                print("loom: no jobs yet")
                return 0
            job_id = rows[0]["id"]
        print(f"watching job {job_id} (ctrl-c to stop)")
        last = 0
        try:
            while True:
                for ev in led.events(job_id, after_id=last):
                    last = ev["id"]
                    ts = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
                    print(f"{ts}  {ev['kind']:10s} {ev['detail']}")
                job = led.job(job_id)
                if job["status"] in ("done", "error") and not led.events(job_id, after_id=last):
                    print(f"job {job_id} finished: {job['status']}")
                    return 0
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0
    finally:
        led.close()


def cmd_say(args):
    led = Ledger(getattr(args, "db", None))
    try:
        job = led.job(args.job) if getattr(args, "job", None) else (led.jobs(limit=1) or [None])[0]
        if not job:
            print("loom: no job to note", file=sys.stderr)
            return 2
        led.add_note(job["id"], args.text)
        print(f"noted for job {job['id']}; the loom folds it into the next dispatch")
        return 0
    finally:
        led.close()


def cmd_stop(args):
    led = Ledger(getattr(args, "db", None))
    try:
        job = led.job(args.job)
        if not job:
            print(f"loom: no job {args.job}", file=sys.stderr)
            return 2
        led.update_job(args.job, status="paused", report="stop requested via omw loom stop")
        # Best-effort: abort every known session so in-flight generations halt.
        try:
            tp = Transport(job["server_url"])
            for t in led.tasks(args.job):
                if t["session_id"]:
                    tp.abort(t["session_id"])
        except LoomServerError:
            pass
        print(f"job {args.job} marked paused; resume with: omw loom resume --job {args.job}")
        return 0
    finally:
        led.close()


def cmd_log(args):
    led = Ledger(getattr(args, "db", None))
    try:
        for ev in led.events(args.job):
            ts = time.strftime("%m-%d %H:%M:%S", time.localtime(ev["ts"]))
            print(f"{ts}  {ev['kind']:10s} {ev['detail']}")
        return 0
    finally:
        led.close()


def cmd_clean(args):
    """Delete finished jobs older than --older-days (default 7) and, when the
    server is reachable, their child sessions."""
    led = Ledger(getattr(args, "db", None))
    try:
        cutoff = time.time() - (getattr(args, "older_days", 7) or 7) * 86400
        removed = 0
        for job in led.jobs(limit=1000):
            if job["status"] not in ("done", "error") or job["updated"] > cutoff:
                continue
            try:
                tp = Transport(job["server_url"], retries=1)
                for t in led.tasks(job["id"]):
                    if t["session_id"]:
                        try:
                            tp.delete_session(t["session_id"])
                        except LoomServerError:
                            pass
            except Exception:  # noqa: BLE001 -- server gone is fine; still drop rows
                pass
            for table in ("tasks", "findings", "notes", "events"):
                led.db.execute(f"DELETE FROM {table} WHERE job_id=?", (job["id"],))
            led.db.execute("DELETE FROM jobs WHERE id=?", (job["id"],))
            led.db.commit()
            removed += 1
        print(f"cleaned {removed} job(s) older than {getattr(args, 'older_days', 7)} day(s)")
        return 0
    finally:
        led.close()


def cmd_pr(args, wire_settings=None, default_models_path=None):
    """Resume the job's implement session to branch/commit/push/open the PR,
    then its review session for PR review. Explicitly human-triggered."""
    led = Ledger(getattr(args, "db", None))
    try:
        job = led.job(args.job)
        if not job:
            print(f"loom: no job {args.job}", file=sys.stderr)
            return 2
        tp = Transport(getattr(args, "attach", None) or job["server_url"])
        loom = Loom(led, tp, args.job, cfg=loom_config(wire_settings),
                    model_ranking=_load_ranking(default_models_path) if default_models_path else {})
        _install_signals(loom)
        parsed, raw = loom._dispatch(
            "agent-code", "implement",
            "The user approved a PR for this work. Create a branch, commit the change with "
            "explicit paths, push, and open the PR. Report the PR URL.", resume=True)
        print(parsed["result"] or raw)
        rparsed, rraw = loom._dispatch(
            "agent-review", "review",
            "A PR was opened for the reviewed work above. Perform PR review per your role "
            "skill.", resume=True)
        print(rparsed["result"] or rraw)
        return 0
    finally:
        led.close()


def _install_signals(loom):
    import signal

    def handler(signum, frame):  # noqa: ARG001
        loom.request_stop()

    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # non-main thread or unsupported platform


def add_parser(sub, io_parent=None):
    """Register `loom` subcommands on omodel-wire.py's argparse subparsers."""
    parents = [io_parent] if io_parent else []
    p = sub.add_parser("loom", parents=parents,
                       help="deterministic pipeline conductor (v2 orchestrator)")
    lsub = p.add_subparsers(dest="loom_cmd", required=True)

    pr = lsub.add_parser("run", help="run a job (called by the loom agent's tool)")
    pr.add_argument("--attach", required=True, help="OpenCode server URL")
    pr.add_argument("--parent", required=True, help="parent (loom agent) session id")
    pr.add_argument("--risk", default="medium", choices=["simple", "medium", "high"])
    pr.add_argument("--packet", help="goal packet (else read from stdin)")
    pr.add_argument("--dir", help="project directory (default: cwd)")
    pr.add_argument("--worker-model", help="run EVERY worker on this model ref")
    pr.add_argument("--json-events", action="store_true")
    pr.add_argument("--db", help=argparse_suppress())

    pask = lsub.add_parser("ask", help="route one question to agent-research (no pipeline)")
    pask.add_argument("--attach", required=True, help="OpenCode server URL")
    pask.add_argument("--parent", required=True, help="parent (loom agent) session id")
    pask.add_argument("--question", help="the question (else read from stdin)")
    pask.add_argument("--dir", help="project directory (default: cwd)")
    pask.add_argument("--worker-model", help="run the research on this model ref")
    pask.add_argument("--json-events", action="store_true")
    pask.add_argument("--db", help=argparse_suppress())

    pres = lsub.add_parser("resume", help="resume a paused job")
    pres.add_argument("--job", type=int, required=True)
    pres.add_argument("--note", help="operator note folded into the next dispatch")
    pres.add_argument("--attach", help="override stored server URL (new port)")
    pres.add_argument("--json-events", action="store_true")
    pres.add_argument("--db", help=argparse_suppress())

    for name, need_job in (("status", False), ("watch", False), ("log", True),
                           ("stop", True), ("pr", True)):
        px = lsub.add_parser(name)
        px.add_argument("--job", type=int, required=need_job)
        px.add_argument("--db", help=argparse_suppress())
        if name == "pr":
            px.add_argument("--attach")

    psay = lsub.add_parser("say", help="queue an operator note for the running job")
    psay.add_argument("text")
    psay.add_argument("--job", type=int)
    psay.add_argument("--db", help=argparse_suppress())

    pclean = lsub.add_parser("clean", help="drop old finished jobs + their sessions")
    pclean.add_argument("--older-days", type=int, default=7)
    pclean.add_argument("--db", help=argparse_suppress())
    return p


def argparse_suppress():
    import argparse
    return argparse.SUPPRESS


def dispatch(args, wire_settings=None, default_models_path=None):
    cmd = args.loom_cmd
    if cmd == "run":
        return cmd_run(args, wire_settings, default_models_path)
    if cmd == "ask":
        return cmd_ask(args, wire_settings, default_models_path)
    if cmd == "resume":
        return cmd_resume(args, wire_settings, default_models_path)
    if cmd == "status":
        return cmd_status(args)
    if cmd == "watch":
        return cmd_watch(args)
    if cmd == "say":
        return cmd_say(args)
    if cmd == "stop":
        return cmd_stop(args)
    if cmd == "log":
        return cmd_log(args)
    if cmd == "clean":
        return cmd_clean(args)
    if cmd == "pr":
        return cmd_pr(args, wire_settings, default_models_path)
    print(f"loom: unknown subcommand {cmd}", file=sys.stderr)
    return 2
