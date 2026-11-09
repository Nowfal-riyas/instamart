# ===== part_d.py: imports and settings =====
import json, re, sqlite3, time, uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Path("outputs").mkdir(exist_ok=True); Path("state").mkdir(exist_ok=True)


# ===== Task 1: chunk documents (40 words, overlap 10) =====
def chunk_documents(folder="data/policies", size=40, overlap=10):
    chunks = []
    for f in sorted(Path(folder).glob("*.txt")):
        words, start = f.read_text(encoding="utf-8").split(), 0
        while True:
            chunks.append({"source": f.name, "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks


# ===== Task 2: TF-IDF retrieval =====
def make_retriever(overlap=10):
    chunks = chunk_documents(overlap=overlap)
    vec = TfidfVectorizer()
    matrix = vec.fit_transform([c["text"] for c in chunks])

    def retrieve(query, k=3):
        sims = cosine_similarity(vec.transform([query]), matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": chunks[i]["source"], "score": float(sims[i]), "text": chunks[i]["text"]} for i in top]
    return chunks, retrieve

chunks, retrieve = make_retriever(10)
print(len(chunks), pd.Series([c["source"] for c in chunks]).value_counts().to_dict())   # 15
print([r["source"] for r in retrieve("refund policy late delivery")])                   # all refund_policy.txt


# ===== Task 3: augment and answer =====
POLICY_SYSTEM = ("You are a POLICY ASSISTANT. Answer only from the supplied context. "
                 "If the answer is not in the context, say you could not find the information.")

def answer(question, retrieve_fn=None):
    retrieve_fn = retrieve_fn or retrieve
    hits = retrieve_fn(question, k=3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    r = mock_llm.chat([{"role": "system", "content": POLICY_SYSTEM},
                       {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}])
    return r.choices[0].message.content, [h["source"] for h in hits]


# ===== Task 4: evaluate retrieval and grounding (overlap 10 and 0) =====
def evaluate(overlap):
    _, ret = make_retriever(overlap)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    has_doc = ev["expected_doc"].astype(str).str.strip().str.endswith(".txt")   # real doc name = answerable
    ans, unans = ev[has_doc], ev[~has_doc]
    doc_ok = fact_ok = 0
    for r in ans.itertuples():
        text, sources = answer(r.question, ret)
        doc_ok += r.expected_doc in sources
        ok = r.expected_fact in text
        fact_ok += ok
        if overlap == 0 and not ok:
            print("  breaks:", r.question)
    fallback = sum(answer(q, ret)[0].startswith("I could not find") for q in unans["question"])
    print(f"overlap={overlap}: doc in top3 {doc_ok}/{len(ans)}, fact in answer {fact_ok}/{len(ans)}, "
          f"fallback {fallback}/{len(unans)}")


# ===== Task 5: tools, spec and tool map =====
LEDGER = "outputs/refund_ledger.jsonl"

def get_order(order_id):
    with sqlite3.connect("data/instamart.db") as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders "
                          "WHERE order_id = ?", (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    dm = None if row[3] is None or row[3] != row[3] else row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": dm}

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    refund = round(order_amount, 2) if delivery_minutes is not None and delivery_minutes > 45 else 0
    return {"refund_inr": refund}

def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}

TOOLS_SPEC = [
    _spec("get_order", "Look up an order by id in the Instamart database.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents for relevant passages.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the refund rules.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue a refund automatically and log it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ===== Tasks 6-8: ReAct loop with limits and checkpoints =====
class SimulatedCrash(Exception):
    pass

AGENT_SYSTEM = "You are a refund AGENT. Use the tools to decide the refund request."
RUNS = "outputs/agent_runs.jsonl"

def _checkpoint(st):
    keys = ("run_id", "scenario_id", "messages", "tokens_used", "step", "executed", "llm_calls", "steps")
    json.dump({k: st[k] for k in keys}, open(f"state/{st['run_id']}.json", "w"), default=str)

def _finish(st, status, answer_text):
    out = {"status": status, "final_answer": answer_text, "steps": st["steps"],
           "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"], "executed": st["executed"]}
    with open(RUNS, "a") as f:
        f.write(json.dumps({"scenario_id": st["scenario_id"], **{k: out[k] for k in
                ("status", "final_answer", "llm_calls", "tokens_used", "steps")}}, default=str) + "\n")
    return out

def _fallback(st, status, why):
    tools = ", ".join(st["executed"]) or "none"
    return _finish(st, status, f"HANDOFF to human: stopped ({why}) after {st['step']} steps; tools run: {tools}.")

def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    while st["llm_calls"] < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback(st, "FALLBACK_TIMEOUT", "time limit")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", "token limit")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return _finish(st, msg.content.split(":")[0].strip(), msg.content)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                               "content": json.dumps(obs, default=str)})
        st["step"] += 1
        st["executed"].append(tc.name)
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        _checkpoint(st)
        if crash_after_tools and len(st["executed"]) == crash_after_tools:
            raise SimulatedCrash(f"crash after {crash_after_tools} tool executions")
    return _fallback(st, "FALLBACK_MAX_ITERATIONS", "max iterations")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"{scenario_id}_{uuid.uuid4().hex[:6]}", "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": AGENT_SYSTEM},
                       {"role": "user", "content": user_message}],
          "tokens_used": 0, "step": 0, "executed": [], "llm_calls": 0, "steps": []}
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    st = json.load(open(f"state/{run_id}.json"))
    return _loop(st, max_iterations, token_limit, time_limit_seconds)


# ===== Detect scenario columns (id, request text, expected refund) =====
scen = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
print(scen.columns.tolist())
id_col = next(c for c in scen.columns if "scenario" in c.lower() or c.lower() == "id")
exp_col = next((c for c in scen.columns if "expected" in c.lower() and "refund" in c.lower()), None)
candidate_cols = [
    c for c in scen.select_dtypes(include=["object", "string"]).columns 
    if "id" not in c.lower()
]
if "message" in scen.columns:
    msg_col = "message"
elif candidate_cols:
    msg_col = max(candidate_cols, key=lambda c: scen[c].astype(str).str.len().mean())
else:
    msg_col = [c for c in scen.columns if c != id_col][0]


# ===== Run all 7 scenarios =====
def refund_of(res):
    vals = [s["observation"].get("refund_inr") for s in res["steps"] if "refund_inr" in s["observation"]]
    return vals[-1] if vals else None

Path(RUNS).unlink(missing_ok=True); Path(LEDGER).unlink(missing_ok=True)
rows = []
for _, r in scen.iterrows():
    res = run_agent(r[msg_col], r[id_col])
    rows.append((r[id_col], res["status"], refund_of(res), res["llm_calls"], res["tokens_used"],
                 r[exp_col] if exp_col else None))
print(pd.DataFrame(rows, columns=["scenario", "status", "refund_inr", "model_calls", "tokens", "expected"]).to_string(index=False))


# ===== Stop-safely demos (S1) =====
s1_msg = scen.iloc[0][msg_col]
print("token_limit=200 :", run_agent(s1_msg, "S1_tok", token_limit=200)["status"])
print("time_limit=0    :", run_agent(s1_msg, "S1_time", time_limit_seconds=0)["status"])


# ===== Crash and resume demo (S1) =====
try:
    run_agent(s1_msg, "S1_crash", run_id="S1_crash", crash_after_tools=2)
except SimulatedCrash as e:
    print("crashed:", e)
res = resume_agent("S1_crash")
print(res["executed"], res["status"], refund_of(res))     # 4 tools, REFUND, 128.25