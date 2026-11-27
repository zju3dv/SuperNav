#!/usr/bin/env python3
"""Run mop-up scan: degeneration patterns + sidecar integrity."""
import json, os, re, sys, glob

if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} RUNS_ROOT VISUALS_ROOT")

runs_root = sys.argv[1]
visuals_root = sys.argv[2]

CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
INFRA_REASON = re.compile(
    r"unreachable|unavailable|policy service|connection refused|timed out|"
    r"became stuck|nonresponsive|KeyError",
    re.IGNORECASE,
)

def scan_run(d):
    issues = []
    mpath = os.path.join(d, "metrics.json")
    if not os.path.isfile(mpath):
        return None  # in-flight, skip
    m = json.load(open(mpath))
    # no_close
    if not m.get("session_closed") or not m.get("close_called"):
        issues.append("no_close")
    if m.get("formal_task_outcome") == "close_failed":
        issues.append("close_failed")
    reason = ((m.get("agent_terminal_claim") or {}).get("reason") or "")
    if INFRA_REASON.search(reason) and (m.get("llm_turns") or 0) < 15:
        issues.append("infra_reason")
    if m.get("returncode") not in (0, None):
        issues.append(f"returncode={m.get('returncode')}")
    cpath = os.path.join(d, "canonical.jsonl")
    malformed = 0
    cjk_claim = False
    last_ts = None
    slow_turns = 0
    from datetime import datetime
    def parse_ts(s):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    if os.path.isfile(cpath):
        for line in open(cpath):
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("type")
            ts = parse_ts(e.get("timestamp") or "")
            if ts and last_ts and ts - last_ts > 600:
                slow_turns += 1
            if ts:
                last_ts = ts
            raw = e.get("raw") or {}
            item = raw.get("item") or {}
            if item.get("type") == "mcp_tool_call":
                args = item.get("arguments")
                if isinstance(args, str):
                    if '"image_ref"' in args:
                        malformed += 1
                    else:
                        try:
                            json.loads(args)
                        except Exception:
                            malformed += 1
            if t == "assistant_text":
                txt = e.get("text") or ""
                if CJK.search(txt) and re.search(r"(?i)(achieved|found|close|success|target)", txt):
                    cjk_claim = True
    if malformed:
        issues.append(f"malformed_call x{malformed}")
    if cjk_claim:
        issues.append("cjk_claim")
    if slow_turns:
        issues.append(f"slow_turn>10min x{slow_turns}")
    # sidecar
    sid = m.get("session_id")
    if sid:
        tj = os.path.join(visuals_root, f"{sid}.trajectory.json")
        if not os.path.isfile(tj):
            issues.append("sidecar_missing")
        else:
            try:
                tr = json.load(open(tj))
                st = tr.get("integrity") or tr.get("capture_status")
                if st != "complete":
                    issues.append(f"sidecar_{st}")
                dense = tr.get("dense_trajectory") or {}
                if len(dense.get("points") or []) < 5:
                    issues.append("sidecar_points<5")
            except Exception as ex:
                issues.append("sidecar_unreadable")
    else:
        issues.append("no_session_id")
    return issues

for d in sorted(glob.glob(os.path.join(runs_root, "*/"))):
    if not os.path.isfile(os.path.join(d, "run.json")):
        continue
    r = scan_run(d.rstrip("/"))
    if r is None:
        print(f"IN_FLIGHT {os.path.basename(d)}")
    elif r:
        print(f"ISSUE {os.path.basename(d)} :: {', '.join(r)}")
print("scan done")
