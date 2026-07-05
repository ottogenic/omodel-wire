#!/usr/bin/env python3
"""
omw_proxy.py -- lightweight debug proxy for OpenCode <-> model traffic.

Stdlib only. Sits in front of the DGX model endpoints, logs the full request and
response body of every call (short request-id per pair), and streams responses
through untouched so OpenCode's SSE UI keeps working.

Routing is by PATH PREFIX: OpenCode's provider baseURL is rewritten to
`http://127.0.0.1:<port>/<route>`, and `proxy_routes.json` maps `<route>` -> the
real upstream baseURL (e.g. http://192.168.50.101:8000/v1). The proxy strips the
route prefix and forwards to `<real baseURL><rest of path>`. This routes every
endpoint (GET /models, POST /chat/completions) to the right model, and lets omw
toggle models on/off independently by editing the map (re-read per request).

Logs land FLAT in the logs dir (default ./proxy_logs), one pair per request:
  <id>_req.json, <id>_res.json, and an append-only index.jsonl.

Run standalone:  python3 utils/omw_proxy.py --port 9099 --logs-dir proxy_logs \
                     --routes ~/.config/opencode/proxy_routes.json
The `omw proxy on|off|replay|read` commands drive it; helpers here are imported by
omodel-wire.py (proxy_logs_dir / find_pair / build_curl / render_read).
"""

import argparse
import json
import os
import secrets
import string
import sys
import time
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

DEFAULT_PORT = 9099
DEFAULT_LOGS_DIRNAME = "proxy_logs"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade"}


# --------------------------------------------------------------------------- #
# Logs + ids (importable)
# --------------------------------------------------------------------------- #
def proxy_logs_dir(base=None):
    """The FLAT logs directory (no date subfolder). Precedence:
    explicit base > $OMW_PROXY_LOGS_DIR > ./proxy_logs. Created if missing."""
    d = base or os.environ.get("OMW_PROXY_LOGS_DIR") or DEFAULT_LOGS_DIRNAME
    d = os.path.expanduser(d)
    os.makedirs(d, exist_ok=True)
    return d


def short_id(logs_dir, n=7):
    """A short (default 7-char) lowercase-alnum id, unique in logs_dir."""
    alphabet = string.ascii_lowercase + string.digits
    for _ in range(200):
        sid = "".join(secrets.choice(alphabet) for _ in range(n))
        if not os.path.exists(os.path.join(logs_dir, f"{sid}_req.json")):
            return sid
    return "".join(secrets.choice(alphabet) for _ in range(n + 4))


def truncate_body(body, max_size=200_000):
    """Cap a huge body for logging. Returns (text, metadata)."""
    if not body:
        return "", {"truncated": False, "length": 0}
    if len(body) <= max_size:
        return body, {"truncated": False, "length": len(body)}
    half = max_size // 2
    return (body[:half] + f"\n\n... [TRUNCATED {len(body) - max_size:,} chars] ...\n\n" + body[-half:],
            {"truncated": True, "length": len(body)})


def _req_path(logs_dir, rid):
    return os.path.join(logs_dir, f"{rid}_req.json")


def _res_path(logs_dir, rid):
    return os.path.join(logs_dir, f"{rid}_res.json")


def find_pair(logs_dir, rid):
    """Return (req_data, res_data|None) for a request-id, or (None, None)."""
    rp = _req_path(logs_dir, rid)
    if not os.path.exists(rp):
        return None, None
    with open(rp, encoding="utf-8") as f:
        req = json.load(f)
    res = None
    sp = _res_path(logs_dir, rid)
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            res = json.load(f)
    return req, res


def load_routes(routes_path):
    """{route: real_baseURL} from proxy_routes.json ({} if absent/bad)."""
    try:
        with open(routes_path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------- #
# curl + human-readable rendering (importable, used by `omw proxy replay/read`)
# --------------------------------------------------------------------------- #
def build_curl(req_data):
    """A copy-pasteable curl for the logged request (hits the real upstream)."""
    method = req_data.get("method", "GET")
    url = req_data.get("url", "")
    headers = req_data.get("headers", {}) or {}
    body = req_data.get("body", "") or ""
    parts = ["curl", f"-X {method}"]
    for k, v in headers.items():
        if k.lower() in ("host", "connection", "content-length", "accept-encoding") or k.lower() in HOP_BY_HOP:
            continue
        parts.append("-H " + _shq(f"{k}: {v}"))
    if body:
        parts.append("-d " + _shq(body))
    parts.append(_shq(url))
    return " ".join(parts)


def _shq(s):
    """Single-quote for POSIX shells."""
    return "'" + s.replace("'", "'\\''") + "'"


class _Ink:
    """ANSI colors, no-op when disabled."""
    def __init__(self, on):
        self.on = on

    def __call__(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def bold(self, s): return self("1", s)
    def dim(self, s): return self("2", s)
    def head(self, s): return self("1;36", s)     # bold cyan section headers
    def role(self, s): return self("1;33", s)     # bold yellow
    def key(self, s): return self("36", s)        # cyan
    def ok(self, s): return self("32", s)
    def err(self, s): return self("31", s)


def _content_to_text(content):
    """OpenAI messages allow content = str OR a list of {type,text/...} parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(p.get("text") or f"[{p.get('type', 'part')}]")
            else:
                out.append(str(p))
        return "\n".join(out)
    return "" if content is None else str(content)


def _assemble_response_text(res_data):
    """Best-effort assistant text from a logged response (JSON or SSE stream)."""
    body = (res_data or {}).get("body", "") or ""
    body = body.strip()
    if not body:
        return "", None
    # non-streaming JSON
    if body.startswith("{"):
        try:
            j = json.loads(body)
            msg = (j.get("choices") or [{}])[0].get("message") or {}
            return _content_to_text(msg.get("content")), msg.get("tool_calls")
        except (json.JSONDecodeError, IndexError, AttributeError):
            pass
    # SSE stream: concatenate choices[].delta.content across `data:` lines
    text, tool_calls = [], []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            delta = (json.loads(payload).get("choices") or [{}])[0].get("delta") or {}
        except (json.JSONDecodeError, IndexError, AttributeError):
            continue
        if delta.get("content"):
            text.append(delta["content"])
        if delta.get("tool_calls"):
            tool_calls.extend(delta["tool_calls"])
    return "".join(text), (tool_calls or None)


def render_read(req_data, res_data, use_color=True):
    """Human-readable, sectioned view of a logged request/response pair."""
    c = _Ink(use_color)
    out = []
    rid = req_data.get("request_id", "?")
    body_raw = req_data.get("body", "") or ""
    try:
        body = json.loads(body_raw) if body_raw.strip().startswith("{") else {}
    except json.JSONDecodeError:
        body = {}

    out.append(c.head(f"=== request {rid} ==="))
    out.append(f"  {c.key('endpoint')} : {req_data.get('method', '?')} {req_data.get('url', '?')}")
    if req_data.get("model") or body.get("model"):
        out.append(f"  {c.key('model')}    : {req_data.get('model') or body.get('model')}")
    # sampling / params (everything except the big structured fields)
    params = {k: v for k, v in body.items() if k not in ("messages", "tools", "model")}
    if params:
        out.append(f"  {c.key('params')}   : " + ", ".join(f"{k}={v}" for k, v in params.items()))

    tools = body.get("tools") or []
    if tools:
        out.append("")
        out.append(c.head(f"--- tools ({len(tools)}) ---"))
        for t in tools:
            fn = (t or {}).get("function", t) or {}
            name = fn.get("name", "?")
            desc = (fn.get("description") or "").strip().splitlines()
            desc = desc[0] if desc else ""
            out.append(f"  {c.bold(name)}: {desc[:100]}")

    messages = body.get("messages") or []
    systems = [m for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]
    if systems:
        out.append("")
        out.append(c.head("--- system ---"))
        for m in systems:
            out.append(_indent(_content_to_text(m.get("content"))))
    if convo:
        out.append("")
        out.append(c.head(f"--- messages ({len(convo)}) ---"))
        for m in convo:
            role = m.get("role", "?")
            out.append(c.role(f"[{role}]"))
            text = _content_to_text(m.get("content"))
            if text:
                out.append(_indent(text))
            for tc in (m.get("tool_calls") or []):
                fn = (tc or {}).get("function", {}) or {}
                out.append(_indent(c.dim(f"-> tool_call {fn.get('name', '?')}({fn.get('arguments', '')})")))

    out.append("")
    if res_data is None:
        out.append(c.head("=== response (none logged) ==="))
    else:
        status = res_data.get("status", "?")
        ms = res_data.get("elapsed_ms", "?")
        tag = c.ok(str(status)) if str(status).startswith("2") else c.err(str(status))
        out.append(c.head(f"=== response {tag} ({ms}ms) ==="))
        text, tool_calls = _assemble_response_text(res_data)
        if text:
            out.append(_indent(text))
        for tc in (tool_calls or []):
            fn = (tc or {}).get("function", {}) or {}
            out.append(_indent(c.dim(f"-> tool_call {fn.get('name', '?')}({fn.get('arguments', '')})")))
        if not text and not tool_calls:
            out.append(_indent(c.dim("(no assistant text parsed; see the raw _res.json)")))
    return "\n".join(out)


def _indent(text, pad="    "):
    return "\n".join(pad + ln for ln in str(text).split("\n"))


# --------------------------------------------------------------------------- #
# The proxy server
# --------------------------------------------------------------------------- #
class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # we log to files, not stderr

    # one entry point for every method
    def do_GET(self): self._handle()
    def do_POST(self): self._handle()
    def do_PUT(self): self._handle()
    def do_DELETE(self): self._handle()
    def do_PATCH(self): self._handle()

    def _fail(self, code, msg):
        payload = json.dumps({"error": msg}).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except OSError:
            pass

    def _handle(self):
        try:
            self._proxy()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass                                     # client hung up (incl. mid-stream write) — nothing to send
        except Exception as e:                       # never let one request kill the thread
            self._fail(502, f"omw-proxy error: {e}")

    def _proxy(self):
        logs_dir = self.server.logs_dir
        routes = load_routes(self.server.routes_path)   # re-read so on/off is live

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        # split /<route>/<rest...> ; map route -> real upstream baseURL
        path = self.path
        seg = path.lstrip("/").split("/", 1)
        route = seg[0] if seg and seg[0] else ""
        rest = "/" + (seg[1] if len(seg) > 1 else "")
        base = routes.get(route)
        if not base:
            self._fail(502, f"no upstream route '{route}' (proxy_routes.json). "
                            f"known: {', '.join(routes) or '(none)'}")
            return
        upstream = base.rstrip("/") + rest

        # upstream headers: drop hop-by-hop, Host (conn sets it), and Accept-Encoding
        # (so the upstream returns identity text we can log/replay verbatim).
        up_headers = {k: v for k, v in self.headers.items()
                      if k.lower() not in HOP_BY_HOP
                      and k.lower() not in ("host", "accept-encoding", "content-length")}

        rid = short_id(logs_dir)
        model = ""
        try:
            model = (json.loads(body).get("model") or "") if body[:1] == b"{" else ""
        except (json.JSONDecodeError, AttributeError):
            pass
        started = time.time()
        _write_json(_req_path(logs_dir, rid),
                    _req_record(rid, route, model, self.command, upstream, up_headers, body))

        u = urlsplit(upstream)
        conn_cls = HTTPSConnection if u.scheme == "https" else HTTPConnection
        conn = conn_cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=600)
        target = u.path + (("?" + u.query) if u.query else "")
        try:
            conn.request(self.command, target, body=body or None, headers=up_headers)
            resp = conn.getresponse()
        except Exception as e:
            self._fail(502, f"upstream {u.hostname}:{u.port} unreachable: {e}")
            return

        resp_headers = resp.getheaders()
        has_len = any(k.lower() == "content-length" for k, _ in resp_headers)
        collected = bytearray()

        self.send_response(resp.status)
        for k, v in resp_headers:
            if k.lower() in HOP_BY_HOP or k.lower() == "content-length":
                continue
            self.send_header(k, v)

        if has_len:
            # non-streaming: read fully, forward with a fresh Content-Length
            payload = resp.read()
            collected += payload
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            # streaming (SSE / chunked): relay chunk-by-chunk, tee into the log
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                collected += chunk
                self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        conn.close()

        elapsed_ms = int((time.time() - started) * 1000)
        _write_json(_res_path(logs_dir, rid),
                    _res_record(rid, resp.status, dict(resp_headers), bytes(collected),
                                elapsed_ms, not has_len))
        _append_index(logs_dir, rid, model, self.command, upstream, resp.status, elapsed_ms)
        print(f"[{rid}] {self.command} {model or route} -> {resp.status} ({elapsed_ms}ms)", flush=True)


def _req_record(rid, route, model, method, url, headers, body):
    text, meta = truncate_body(body.decode("utf-8", "replace"))
    return {"request_id": rid, "timestamp": time.time(), "route": route, "model": model,
            "method": method, "url": url, "headers": headers, "body": text, "body_metadata": meta}


def _res_record(rid, status, headers, body, elapsed_ms, streamed):
    text, meta = truncate_body(body.decode("utf-8", "replace"))
    return {"request_id": rid, "timestamp": time.time(), "status": status, "headers": headers,
            "body": text, "body_metadata": meta, "elapsed_ms": elapsed_ms, "streamed": streamed}


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _append_index(logs_dir, rid, model, method, url, status, ms):
    line = json.dumps({"id": rid, "ts": time.time(), "model": model, "method": method,
                       "url": url, "status": status, "ms": ms})
    with open(os.path.join(logs_dir, "index.jsonl"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_server(port, logs_dir, routes_path):
    logs_dir = proxy_logs_dir(logs_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), ProxyHandler)
    server.daemon_threads = True
    server.logs_dir = logs_dir
    server.routes_path = os.path.expanduser(routes_path)
    print(f"omw-proxy on http://127.0.0.1:{port}  logs={logs_dir}  routes={server.routes_path}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    ap = argparse.ArgumentParser(description="omw-proxy -- debug proxy for OpenCode model traffic")
    ap.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    ap.add_argument("--logs-dir", "-l", default=DEFAULT_LOGS_DIRNAME)
    ap.add_argument("--routes", "-r", default="proxy_routes.json",
                    help="path to the {route: baseURL} map")
    args = ap.parse_args()
    run_server(args.port, args.logs_dir, args.routes)


if __name__ == "__main__":
    main()
