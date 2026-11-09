# ===== part_c.py: imports and settings =====
import json, random, re, time
from pathlib import Path
import pandas as pd
import mock_llm

# ASSUMED mock_llm interface (edit here if yours differs):
est_tokens = mock_llm.est_tokens                 # est_tokens(text) -> int
TransientLLMError = mock_llm.TransientLLMError
def call_model(system_prompt, user_message):
    r = mock_llm.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])
    return r.choices[0].message.content, r.usage.total_tokens

MAX_INPUT_TOKENS, MAX_RETRIES, BASE_DELAY = 400, 3, 0.05
CATS = {"billing", "delivery", "technical", "fraud"}
Path("outputs").mkdir(exist_ok=True)
# ASSUMED ticket columns: ticket_id, customer_id, text, true_category
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)


# ===== Redaction (order: email, phone, order id) =====
PATTERNS = [("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.]+"),
            ("PHONE", r"\b[6-9]\d{9}\b"),
            ("ORDER_ID", r"ORD-\d{6}")]

def redact(text, counts=None):
    for name, pat in PATTERNS:
        text, n = re.subn(pat, f"[{name}]", text)
        if counts is not None:
            counts[name] = counts.get(name, 0) + n
    return text

counts = {}
redacted_all = [redact(t, counts) for t in tickets["text"]]
print(counts, "total:", sum(counts.values()))          # 55, 91, 132, 278
assert all(not re.search(p, t) for t in redacted_all for _, p in PATTERNS)
print("left after re-applying patterns: 0")


# ===== Prompts =====
FORMAT = ('Reply with JSON only: {"category": "...", "reason": "..."}. '
          "Allowed categories: billing, delivery, technical, fraud.")
ZERO_SHOT = "CLASSIFY the customer support ticket. " + FORMAT
FEW_SHOT = ("CLASSIFY the customer support ticket. " + FORMAT + "\nEXAMPLES:\n"
    'Ticket: I was charged twice for my order.\n{"category": "billing", "reason": "double charge"}\n'
    'Ticket: My order never arrived.\n{"category": "delivery", "reason": "missing order"}\n'
    'Ticket: The app crashes when I open it.\n{"category": "technical", "reason": "app crash"}\n'
    'Ticket: Someone used my account to place orders I did not make.\n{"category": "fraud", "reason": "unauthorised use"}')
assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT


# ===== Retry wrapper (exponential backoff with jitter) =====
transient_errors = 0

def call_with_retry(system_prompt, message, ticket_id):
    global transient_errors
    for retry in range(MAX_RETRIES + 1):
        try:
            return call_model(system_prompt, message)
        except TransientLLMError:
            transient_errors += 1
            if retry == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            print(f"retry ticket={ticket_id} retry_no={retry} wait={wait:.4f}s")
            time.sleep(wait)


# ===== Output validation =====
def validate(text):
    try:
        d = json.loads(text)
        if isinstance(d, dict) and d.get("category") in CATS:
            return "classified_json", d["category"]
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        found = set(re.findall(r"[a-z]+", str(text).lower())) & CATS
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", None


# ===== Pipeline: guardrails -> redact -> call -> validate =====
def run_pipeline(system_prompt, token_budget=None):
    global transient_errors
    mock_llm.reset()
    transient_errors, total, rows = 0, 0, []
    for t in tickets.itertuples():
        row = dict(ticket_id=t.ticket_id, customer_id=t.customer_id,
                   true_category=t.true_category, category=None, tokens_used=0)
        if t.text.strip() == "":
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            msg = redact(t.text)
            assert not any(re.search(p, msg) for _, p in PATTERNS)   # no PII in prompt
            reply, used = call_with_retry(system_prompt, msg, t.ticket_id)
            total += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate(reply)
        rows.append(row)
    return pd.DataFrame(rows)

CLASSIFIED = ["classified_json", "classified_keyword"]

def summarise(name, df):
    c = df[df["status"].isin(CLASSIFIED)]
    acc = (c["category"] == c["true_category"]).mean()
    print(f"{name}: classified={len(c)} accuracy={acc:.1%} total_tokens={df['tokens_used'].sum()} "
          f"per_ticket={df['tokens_used'].sum() / len(c):.1f} transient_errors={transient_errors}")
    print(df["status"].value_counts().to_dict())


# ===== Zero-shot vs few-shot (no budget) =====
zero = run_pipeline(ZERO_SHOT);  summarise("zero-shot", zero)      # 135/12/9, 85.7%
few = run_pipeline(FEW_SHOT);    summarise("few-shot", few)        # 100%
zc = zero[zero["status"].isin(CLASSIFIED)]
print(pd.crosstab(zc["true_category"], zc["category"]))            # zero-shot confusion table


# ===== Budget demo (zero-shot) =====
BUDGET_DEMO = 6000                                                  # lower to 5000 if nothing is exhausted
demo = run_pipeline(ZERO_SHOT, token_budget=BUDGET_DEMO)
demo.to_csv("outputs/triage_budget_demo.csv", index=False)
print("budget", BUDGET_DEMO, "exhausted:", (demo["status"] == "budget_exhausted").sum())


# ===== Final few-shot run with routing =====
final = run_pipeline(FEW_SHOT, token_budget=100000)
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
final["segment_name"] = final["customer_id"].map(seg["segment_name"])
final["is_premium"] = final["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)

def route(r):
    if r["status"] in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if r["category"] == "fraud":
        return "security_team", "HIGH"
    if r["is_premium"]:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

final[["route", "priority"]] = final.apply(lambda r: pd.Series(route(r)), axis=1)
final[["ticket_id", "customer_id", "status", "category", "segment_name",
       "priority", "route", "tokens_used"]].to_csv("outputs/triage_results.csv", index=False)
print(len(final), final["route"].value_counts().to_dict())          # 164 rows