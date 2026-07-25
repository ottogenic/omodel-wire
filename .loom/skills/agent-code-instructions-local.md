# omodel-wire Coding Overlay

## Repo Invariants

- Stdlib only; the implementation stays in  plus tests ( holds
  the loom conductor and proxy helpers).
- omodel-manager owns model configs; never add  or .
- Preserve idempotent, non-destructive config merging and cross-platform paths.
- Keep LF endings,  plural, and  on every model.
- Add an  changelog entry for user-visible behavior.
- Changes to , /tags, , or security-sensitive paths need
  explicit user approval.

## Focused Check

Run  plus the targeted   note: model configs dir not found: /tmp/tmpgebvtkzd/nope
        point it with --configs PATH or $OMODEL_CONFIGS (omodel-manager's configs/).
[0u9jiiv] GET dgx-n1-8000 -> 200 (1ms)
[edoemss] POST qwen/qwen3.6-35b -> 200 (1ms)
{"type": "start", "detail": "ask job 1", "title": "ask job 1"}
{"type": "session", "detail": "agent-research [ask] -> ses_1", "title": "agent-research [ask] -> ses_1", "session": "ses_1"}
{"type": "compose", "detail": "agent-research: no instruction files", "title": "agent-research: no instruction files", "session": "ses_1"}
{"type": "dispatch", "detail": "agent-research [ask]", "title": "ask: agent-research working...", "session": "ses_1"}
{"type": "status", "detail": "agent-research [ask] -> DONE", "title": "ask: agent-research -> DONE"}
{"type": "done", "detail": "research answered", "title": "answered"}
{"type": "report", "report": "LOOM RESEARCH ANSWER

Bananas: rich in potassium [usda.gov]

EVIDENCE/SOURCES:
usda.gov/banana, healthline.com/banana"}
cleaned 1 job(s) older than 1 day(s)
{"type": "start", "detail": "job 1 risk=simple parent=ses_parent", "title": "job 1 risk=simple parent=ses_parent"}
{"type": "phase", "detail": "plan skipped (risk=simple)", "title": "plan skipped (risk=simple)"}
{"type": "phase", "detail": "code: agent-code", "title": "coding..."}
{"type": "session", "detail": "agent-code [implement] -> ses_1", "title": "agent-code [implement] -> ses_1", "session": "ses_1"}
{"type": "compose", "detail": "agent-code: no instruction files", "title": "agent-code: no instruction files", "session": "ses_1"}
{"type": "dispatch", "detail": "agent-code [implement]", "title": "implement: agent-code working...", "session": "ses_1"}
{"type": "status", "detail": "agent-code [implement] -> BLOCKED", "title": "implement: agent-code -> BLOCKED"}
{"type": "session", "detail": "agent-architect [diagnose] -> ses_2", "title": "agent-architect [diagnose] -> ses_2", "session": "ses_2"}
{"type": "compose", "detail": "agent-architect: no instruction files", "title": "agent-architect: no instruction files", "session": "ses_2"}
{"type": "dispatch", "detail": "agent-architect [diagnose]", "title": "diagnose: agent-architect working...", "session": "ses_2"}
{"type": "status", "detail": "agent-architect [diagnose] -> BLOCKED", "title": "diagnose: agent-architect -> BLOCKED"}
{"type": "escalate", "detail": "agent-code [implement]: fresh session, distilled packet (blocked and architect could not diagnose)", "title": "agent-code [implement]: fresh session, distilled packet (blocked and architect could not diagnose)"}
{"type": "session", "detail": "agent-code [implement] -> ses_3", "title": "agent-code [implement] -> ses_3", "session": "ses_3"}
{"type": "compose", "detail": "agent-code: no instruction files", "title": "agent-code: no instruction files", "session": "ses_3"}
{"type": "dispatch", "detail": "agent-code [implement]", "title": "implement: agent-code working...", "session": "ses_3"}
{"type": "status", "detail": "agent-code [implement] -> BLOCKED", "title": "implement: agent-code -> BLOCKED"}
{"type": "dispatch", "detail": "agent-architect [diagnose] (resume)", "title": "diagnose: agent-architect working...", "session": "ses_2"}
{"type": "status", "detail": "agent-architect [diagnose] -> BLOCKED", "title": "diagnose: agent-architect -> BLOCKED"}
{"type": "escalate", "detail": "agent-code [implement]: retry on next-ranked model (blocked and architect could not diagnose)", "title": "agent-code [implement]: retry on next-ranked model (blocked and architect could not diagnose)"}
{"type": "session", "detail": "agent-code [implement] -> ses_4", "title": "agent-code [implement] -> ses_4", "session": "ses_4"}
{"type": "compose", "detail": "agent-code: no instruction files", "title": "agent-code: no instruction files", "session": "ses_4"}
{"type": "dispatch", "detail": "agent-code [implement]", "title": "implement: agent-code working...", "session": "ses_4"}
{"type": "status", "detail": "agent-code [implement] -> BLOCKED", "title": "implement: agent-code -> BLOCKED"}
{"type": "dispatch", "detail": "agent-architect [diagnose] (resume)", "title": "diagnose: agent-architect working...", "session": "ses_2"}
{"type": "status", "detail": "agent-architect [diagnose] -> BLOCKED", "title": "diagnose: agent-architect -> BLOCKED"}
{"type": "escalate", "detail": "agent-code [implement]: retry on next-ranked model (blocked and architect could not diagnose)", "title": "agent-code [implement]: retry on next-ranked model (blocked and architect could not diagnose)"}
{"type": "session", "detail": "agent-code [implement] -> ses_5", "title": "agent-code [implement] -> ses_5", "session": "ses_5"}
{"type": "compose", "detail": "agent-code: no instruction files", "title": "agent-code: no instruction files", "session": "ses_5"}
{"type": "dispatch", "detail": "agent-code [implement]", "title": "implement: agent-code working...", "session": "ses_5"}
{"type": "status", "detail": "agent-code [implement] -> BLOCKED", "title": "implement: agent-code -> BLOCKED"}
{"type": "dispatch", "detail": "agent-architect [diagnose] (resume)", "title": "diagnose: agent-architect working...", "session": "ses_2"}
{"type": "status", "detail": "agent-architect [diagnose] -> BLOCKED", "title": "diagnose: agent-architect -> BLOCKED"}
{"type": "paused", "detail": "agent-code [implement] stuck (blocked and architect could not diagnose). Resume with: omw loom resume --job 1", "title": "paused: agent-code needs a human"}
{"type": "paused", "detail": "job 1 paused; resume with: omw loom resume --job 1", "title": "job 1 paused; resume with: omw loom resume --job 1"}
{"type": "start", "detail": "ask job 1", "title": "ask job 1"}
{"type": "session", "detail": "agent-research [ask] -> ses_1", "title": "agent-research [ask] -> ses_1", "session": "ses_1"}
{"type": "compose", "detail": "agent-research: no instruction files", "title": "agent-research: no instruction files", "session": "ses_1"}
{"type": "dispatch", "detail": "agent-research [ask]", "title": "ask: agent-research working...", "session": "ses_1"}
{"type": "malformed", "detail": "agent-research reply missing STATUS; nudging for the contract block", "title": "agent-research reply missing STATUS; nudging for the contract block"}
{"type": "status", "detail": "agent-research [ask] -> DONE", "title": "ask: agent-research -> DONE"}
{"type": "done", "detail": "research answered", "title": "answered"}
{"type": "report", "report": "LOOM RESEARCH ANSWER

skill file has 128 lines

EVIDENCE/SOURCES:
read SKILL.md"} for the area you changed. Use , not . The full suite
belongs to .
