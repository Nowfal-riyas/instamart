Instamart — End-to-End Data Analytics, Machine Learning & AI Support Automation
Overview

This project develops an end-to-end data analytics and AI-powered support automation system for Instamart.

The project covers four major areas:

Part A — Data Foundation: Auditing, cleaning, storing and analysing raw customer, product and order data using Python, Pandas and SQLite.
Part B — Customer Segmentation & Lapse Prediction: Applying RFM analysis, K-Means clustering, PCA and machine learning models to identify customer segments and predict customer inactivity.
Part C — Support Ticket Triage: Building a privacy-aware ticket classification pipeline with PII redaction, retry handling, output validation, token budgeting and customer-based routing.
Part D — Policy RAG Assistant & Refund Agent: Developing a TF-IDF-based retrieval system and a tool-using refund agent with policy grounding, execution limits and resumable checkpoints.

All model interactions use the supplied mock_llm.py. No real external API calls or API keys are required.

1. Clone the repository
bash
git clone https://github.com/Nowfal-riyas/instamart.git
cd instamart

2. Create a virtual environment
bash
python -m venv venv


Activate it on Windows PowerShell:

powershell
.\venv\Scripts\activate

3. Install dependencies
bash
pip install -r requirements.txt

4. Generate the raw datasets
bash
python generate_data.py


The supplied input files and generated datasets are used throughout the project.

5. Run the project

Run the scripts in dependency order:

python generate.py
python audit.py
python model.py
python support_ticket.py
python policy_assistant.py


Adjust script paths to match the actual filenames in the repository.

The scripts generate cleaned datasets, the SQLite database, SQL analysis results, customer segments, trained models, ticket classifications and refund-agent execution logs.

Part A — Data Foundation: Clean, Store and Query
1. Raw Data Audit

The three raw datasets were loaded using Pandas.

The ticket dataset was loaded with:

python
pd.read_csv("data/raw/tickets.csv", keep_default_na=False)


This preserves empty ticket texts as empty strings.

The audit included:

Dataset shapes.
Column data types.
Missing-value counts and percentages.
Exact duplicate rows.
Invalid numeric values.
Invalid delivery times.
Inconsistent customer city and gender spellings.
Raw Audit Results
Audit metric	Result
Raw order rows	13,084
Exact duplicate rows	119
Pending order rows	656
Distinct order IDs	12,309
Non-numeric order amounts	120
Negative delivery times	106
Blank delivery times	2,541
Blank customer ages	61
Ages outside 10–100	6
Distinct raw city spellings	24
Distinct non-blank gender spellings	10
Blank genders	30
2. Customer Data Cleaning

The following cleaning operations were applied:

City values were stripped of leading and trailing whitespace and converted to title case.
City spellings were standardised into six cities: Pune, Delhi, Bengaluru, Chennai, Hyderabad and Mumbai.
Gender spellings were mapped to male, female or unknown.
Blank gender values were mapped to unknown.
Ages outside the range 10–100 were converted to missing values.
Missing ages were filled using the median age of the corresponding city, calculated with groupby() and transform().
Signup dates were converted to datetime values.

Customer cleaning results:

City	Customers
Pune	227
Delhi	210
Bengaluru	197
Chennai	191
Hyderabad	191
Mumbai	184
Gender	Customers
Female	610
Male	560
Unknown	30

After cleaning, there were zero missing ages and all customer ages were between 18 and 60.

3. Order Data Cleaning

Cleaning was performed in the following order:

Converted order_ts into datetime format.
Removed exact duplicate rows (119 rows).
For duplicate order_id values with different statuses, retained the record with the latest order_ts.
Converted order_amount into numeric values, replacing unparseable entries with missing values.
Filled missing order amounts using:

round(price_inr × quantity × (1 − discount_pct / 100), 2)

Converted negative delivery_minutes values into missing values.
Filled missing delivery times for completed orders using the median delivery time of the corresponding product category among completed orders.
Left cancelled-order delivery times missing because cancelled orders were never delivered.
4. Final Cleaned Data
Metric	Result
Final orders	12,309
Completed orders	11,110
Cancelled orders	1,199
Pending orders	0
Missing order amounts	0
Missing delivery times in completed orders	0
Missing delivery times in cancelled orders	1,199
Completed order revenue	INR 2,998,668.80
5. SQLite Database

The cleaned datasets were saved as CSV files and stored in:

data/instamart.db

The database contains three tables:

customers
products
orders

The order_ts column is stored in YYYY-MM-DD HH:MM:SS format.

A validation script checks the reference checkpoints and prints:

text
ALL CHECKPOINTS PASSED

6. SQL Business Analysis

Seven SQL queries were implemented in sql/queries.sql and executed using pd.read_sql_query().

Query	Analysis	Expected Result
Q1	WHERE and HAVING: high-value customers	30 customers
Q2	Three-table join: revenue by city and category	30 rows
Q3	Scalar subquery: products above average price	11 products
Q4	LEFT JOIN: customers without completed orders	25 customers
Q5	CTE: customers spending above average	550 customers
Q6	Ranking window functions	8 rank/row-number differences
Q7	Running daily revenue total	183 daily rows

Important SQL findings:

Total completed revenue: INR 2,998,668.80.
Highest city-category revenue: Bengaluru / Electronics, INR 233,909.45.
Average customer completed spend: INR 2,552.06.
Final cumulative revenue: INR 2,998,668.80.

In Q6, ties can cause RANK() to assign the same rank to multiple customers, whereas ROW_NUMBER() always assigns a unique sequential number.

7. Pandas Cross-Check and Exploratory Analysis

Q2 was independently reconstructed using pd.pivot_table() with margins=True. Its grand total matched the SQL revenue total.

City-level completed-order analysis:

City	Revenue	Mean Order Amount	Order Count
Bengaluru	INR 523,643.40	INR 276.47	1,894
Mumbai	INR 479,058.50	INR 297.00	1,613

Completed order amount distribution:

Statistic	Value
Mean	INR 269.91
Median	INR 135.00
Skewness	4.61
Q1	INR 64.80
Q3	INR 297.00
Upper IQR fence	INR 645.30
Outliers above upper fence	897
Outliers below lower fence	0

The completed order amounts are strongly right-skewed, with the mean substantially higher than the median. There are 897 upper-tail outliers and no lower-tail outliers. These values reflect a dataset containing many smaller purchases alongside a smaller number of expensive orders.

The correlation analysis showed that order_amount and discount_pct had the largest absolute correlation, approximately -0.095. The correlation between order_amount and quantity was approximately +0.087. These relationships are weak, and a correlation of this size does not establish causation.

Part B — Customer Segments and Lapse Prediction
1. RFM Analysis

RFM analysis was performed using completed orders and the snapshot date:

2025-10-01 00:00:00

The three metrics were:

Recency: Days since the customer's most recent completed order.
Frequency: Number of distinct completed orders.
Monetary: Total completed order spending.

RFM summary:

Metric	Result
Customers	1,175
Median recency	19 days
Minimum recency	0 days
Maximum recency	180 days
Mean frequency	9.46
Maximum frequency	33
Mean monetary	INR 2,552.06
Maximum monetary	INR 14,766.55
2. K-Means Clustering

The RFM features were standardised using StandardScaler.

K-Means was evaluated for K values from 2 to 8 using:

n_init=10
random_state=42
Inertia
Silhouette score
K	Inertia	Silhouette Score
2	1820.8	0.465
3	1268.2	0.445
4	743.1	0.526
5	617.4	0.481
6	503.2	0.482
7	408.5	0.416
8	366.5	0.412

The final selected value was K = 4.

Although inertia continually decreases as K increases, adding more clusters naturally reduces within-cluster distances. Therefore, inertia alone cannot identify the best number of clusters; silhouette quality, the elbow and business interpretability must also be considered.

3. Segment Profiling and Business Actions

The four clusters were profiled using their mean recency, frequency and monetary values.

Segment	Customers	Business Name	Recommended Action
Segment 1	436	[YOUR SEGMENT NAME]	[YOUR ACTION]
Segment 2	138	[YOUR SEGMENT NAME]	[YOUR ACTION]
Segment 3	419	[YOUR SEGMENT NAME]	[YOUR ACTION]
Segment 4	182	[YOUR SEGMENT NAME]	[YOUR ACTION]

The 138-customer segment had approximately INR 7,130 mean monetary value and 6.8 mean frequency.

The 419-customer segment had approximately 19.4 mean frequency and 8.8 days mean recency.

The segment with the highest mean monetary value was marked as Premium.

Cluster IDs are arbitrary labels and do not indicate a ranking or customer value.

The resulting segment assignments were saved to:

outputs/customer_segments.csv

Columns:

customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium

4. PCA

PCA was performed on the scaled RFM features using two components.

Component	Explained Variance
PC1	0.634
PC2	0.218
Total	0.852

The first two principal components explain approximately 85.2% of the variance in the scaled RFM data.

5. Lapse Prediction Dataset

The lapse prediction cutoff was:

2025-09-01

The cohort included customers with at least one completed order before the cutoff.

A customer was labelled as lapsed if they had no completed order between September 1 and September 30, 2025.

Features included:

Recency
Frequency
Monetary
Average discount percentage
Cancellation rate
Average delivery time
Favourite product category
City
Age
Gender
Customer tenure

No orders dated on or after September 1, 2025 were used to calculate model features.

Segment labels from Part B were not used as model features.

Cohort results:

Metric	Result
Cohort customers	1,156
Lapsed customers	443
Lapsed percentage	38.32%
Training rows	924
Test rows	232

The train-test split used stratification and random_state=42.

6. Preprocessing Without Data Leakage

Numeric features were standardised, while nominal features were one-hot encoded using drop="first".

All preprocessing was fitted using training data only and then applied to the test data.

No encoder, scaler or resampler saw test rows during fitting.

Missing feature values were checked and confirmed to be absent.

Frequency Statistic	Value
Full-cohort mean	[YOUR RESULT]
Training scaler mean	[YOUR RESULT]

These values differ because the scaler was fitted on the training subset rather than the entire cohort.

7. Model Training and Evaluation

The baseline model always predicted that customers would not lapse.

Baseline performance:

Accuracy: 61.6%
Lapsed-class F1: 0

Two models were tuned using GridSearchCV with five-fold cross-validation and F1 scoring.

Logistic Regression

Model: LogisticRegression(max_iter=1000)
C values: 0.01, 0.1, 1, 10, 100
Best parameters: [YOUR RESULT]
Best cross-validation F1: [YOUR RESULT]

Random Forest

Model: RandomForestClassifier(n_estimators=100, random_state=42)
Maximum depth: 3, 5, 8
Minimum samples per leaf: 5, 10, 20
Total combinations: 9
Total cross-validation fits: 45
Best parameters: [YOUR RESULT]
Best cross-validation F1: [YOUR RESULT]
8. Test Set Evaluation
Metric	Logistic Regression	Random Forest
Precision	[YOUR RESULT]	[YOUR RESULT]
Recall	[YOUR RESULT]	[YOUR RESULT]
F1 Score	[YOUR RESULT]	[YOUR RESULT]

Confusion matrices:

Logistic Regression

text
[INSERT ACTUAL CONFUSION MATRIX]


Random Forest

text
[INSERT ACTUAL CONFUSION MATRIX]


The selected model was [YOUR SELECTED MODEL] because [YOUR REASON].

Missing a customer who is about to lapse can result in lost future revenue and a missed retention opportunity. However, incorrectly targeting loyal customers wastes campaign resources and may create unnecessary promotional costs. The selected model balances these competing business costs based on its precision and recall.

9. Class-Balance Experiment

The training data was oversampled by randomly duplicating minority-class rows with replacement until both classes had equal counts.

The test set remained unchanged.

Metric	Original Logistic Regression	Balanced Logistic Regression
Precision	[YOUR RESULT]	[YOUR RESULT]
Recall	[YOUR RESULT]	[YOUR RESULT]
F1 Score	[YOUR RESULT]	[YOUR RESULT]

Recall increased from approximately 0.64 to 0.85 in the reference experiment, while precision remained near 0.60.

This means the retention team can identify more genuinely lapsing customers, although more loyal customers may also receive unnecessary retention campaigns.

10. Feature Importance and Model Persistence

The five most important Random Forest features were:

[YOUR FEATURE]
[YOUR FEATURE]
[YOUR FEATURE]
[YOUR FEATURE]
[YOUR FEATURE]

Three Logistic Regression coefficients:

Feature	Coefficient Sign	Business Interpretation
[FEATURE]	[Positive/Negative]	[INTERPRETATION]
[FEATURE]	[Positive/Negative]	[INTERPRETATION]
[FEATURE]	[Positive/Negative]	[INTERPRETATION]

The selected model was saved to:

outputs/lapse_model.joblib

The saved model was reloaded using joblib.load() and its predictions were compared against the original test predictions.

The predictions matched exactly.

11. Premium Customer Retention List

The model's predicted lapse probabilities were used to rank Premium customers in the test set.

The top 10 customers, or all eligible customers if fewer than 10, were saved to:

outputs/retention_list.csv

Columns:

customer_id, lapse_probability, actually_lapsed

Number of customers in the retention list who actually lapsed: [YOUR RESULT].

Important limitation: The Premium flag is derived from whole-window RFM data that includes the September lapse-label period. Therefore, the retention list is illustrative and is not a leakage-free evaluation.

Part C — Support Ticket Triage with Guardrails
1. Input Guardrails

The raw support dataset contained 164 tickets.

Before any model call, two checks were applied:

Empty or whitespace-only tickets were marked skipped_empty.
Tickets exceeding 400 estimated tokens were marked flagged_long.
Status	Count
Empty tickets	4
Long tickets	4
Eligible tickets	156

Neither empty nor long tickets were sent to the model.

2. Personally Identifiable Information Redaction

The following regular expressions were applied in the required order:

Information	Pattern	Placeholder
Email	[\w.+-]+@[\w-]+\.[\w.]+	[EMAIL]
Phone	\b[6-9]\d{9}\b	[PHONE]
Order ID	ORD-\d{6}	[ORDER_ID]

Redaction results:

PII Type	Substitutions
Emails	55
Phone numbers	91
Order IDs	132
Total	278

A second validation pass confirmed that zero matching PII patterns remained in the redacted texts.

Only the redacted ticket text was sent as the user message to the model.

3. Classification Prompts

Two system prompts were implemented:

Zero-shot classification.
Few-shot classification.

Both prompts required the model to classify tickets into:

billing
delivery
technical
fraud

The model was instructed to return JSON containing category and reason.

The few-shot prompt included labelled examples, including a fraud example.

4. Retry Handling

Transient model errors were handled using exponential backoff with jitter.

Configuration:

Maximum retries: 3 per ticket.
Base delay: 0.05 seconds.
Jitter: random.uniform(0.9, 1.15).
Retry delay: BASE_DELAY × 2**n × jitter.

The mock model was reset immediately before each full run.

The full run generated 19 transient errors, and all were recovered through retries.

5. Output Validation

Every model response was validated.

Output Status	Count
classified_json	135
classified_keyword	12
flagged_invalid_output	9

Valid JSON responses were accepted only if the category was one of the four allowed categories.

For non-JSON responses, lowercase alphabetic words were extracted using:

re.findall(r"[a-z]+", text.lower())

A response was accepted only if exactly one allowed category appeared.

All other responses were flagged for human review.

6. Zero-Shot vs Few-Shot Comparison
Metric	Zero-Shot	Few-Shot
Classified tickets	147	147
Accuracy	85.7%	100%
Correct classifications	126	147
Total tokens	[YOUR RESULT]	[YOUR RESULT]
Tokens per classified ticket	[YOUR RESULT]	[YOUR RESULT]

Zero-shot confusion table:

text
[INSERT YOUR ACTUAL CONFUSION TABLE]


The zero-shot model achieved 85.7% accuracy, while few-shot classification achieved 100% accuracy in the experiment.

The few-shot prompt required additional tokens, but the improvement was substantial. In this experiment, the additional prompt cost was worthwhile because it eliminated the observed classification errors and reduced the risk of incorrect ticket routing.

7. Token Budget and Routing

The zero-shot demonstration used a token budget of [YOUR BUDGET].

Tickets that reached the budget limit were marked:

budget_exhausted

Number of budget-exhausted tickets: [YOUR RESULT].

The demonstration output was saved to:

outputs/triage_budget_demo.csv

8. Final Ticket Routing

The final pipeline used the few-shot prompt with a token budget of 100,000.

Routing rules:

Condition	Route	Priority
Empty, long, invalid or budget exhausted	human_review	NORMAL
Fraud category	security_team	HIGH
Premium customer	premium_desk	HIGH
All other classified tickets	support_queue	NORMAL

Customers without a matching segment record were treated as non-premium.

Final output:

outputs/triage_results.csv

The file contains 164 rows and the following columns:

ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used

Rule-validated classifications were routed automatically according to the routing rules. Empty, long, invalid and budget-exhausted tickets were routed to human review.

Part D — Policy RAG Assistant and Refund Agent
1. Policy Document Chunking

Five policy documents were loaded from data/policies/.

Documents were split into 40-word chunks with an overlap of 10 words.

Policy Document	Chunks
refund_policy.txt	4
delivery_policy.txt	3
escalation_policy.txt	3
payment_policy.txt	3
privacy_policy.txt	2
Total	15

Each chunk retained its source filename.

2. TF-IDF Retrieval

A TfidfVectorizer() was fitted on the chunk texts.

Queries were transformed using the same vectorizer, and cosine similarity was used to rank the chunks.

The retrieval function:

retrieve(query, k=3)

returns dictionaries containing:

source
score
text

For the query refund policy late delivery, the top three retrieved chunks came from refund_policy.txt.

TF measures how frequently a term occurs in a document, while IDF reduces the weight of terms that occur frequently across documents. A limitation of TF-IDF is that it relies on lexical overlap, so two semantically equivalent phrases with different words may not retrieve each other effectively.

3. Policy Assistant

The policy assistant retrieves three relevant chunks and sends them to the mock model.

The system prompt contains POLICY ASSISTANT and instructs the model to:

Answer only using the supplied policy context.
Avoid inventing policy information.
Return the fallback response when the context does not contain the answer.

The user message contains the retrieved context and the original question.

The function returns the model answer and the sources used.

4. Retrieval and Grounding Evaluation

The evaluation dataset contained 14 questions:

12 answerable questions.
2 questions not covered by the policy documents.

Evaluation results:

Chunking Strategy	Expected Document Retrieved	Expected Fact Found	Unanswerable Fallbacks
40 words, overlap 10	12/12	12/12	2/2
40 words, overlap 0	[YOUR RESULT]/12	[YOUR RESULT]/12	[YOUR RESULT]/2

With overlap, all answerable questions retrieved the expected document and produced answers containing the expected fact.

Without overlap, [QUESTION NAME] failed because the relevant information was split across a chunk boundary, leaving the retrieval system without sufficient context.

5. Refund Agent Tools

Five tools were implemented and registered in TOOL_MAP.

Tool	Purpose
get_order	Retrieves order details from SQLite
search_policy	Retrieves relevant policy chunks
calculate_refund	Calculates the eligible refund amount
issue_refund	Records an automatically approved refund
escalate_to_human	Records a refund requiring human approval

Refund rules:

Only completed orders are eligible.
Delivery time greater than 45 minutes qualifies for a 100% refund.
Delivery time of 45 minutes or less qualifies for INR 0 refund.
Refunds above INR 1,000 must be escalated to a human.
All simulated requests are assumed to be within the seven-day policy window.

Refund and escalation actions were recorded in:

outputs/refund_ledger.jsonl

6. ReAct Agent Execution

The refund agent uses a ReAct loop consisting of:

Model reasoning.
Tool selection.
Tool execution.
Tool observation.
Further reasoning or final response.

Agent configuration:

Maximum iterations: 6.
Token limit: 6,000.
Time limit: 10 seconds.

The agent records:

Execution status.
Final answer.
Tool steps.
Number of model calls.
Tokens used.
7. Scenario Results

All seven supplied scenarios were executed.

Scenario	Expected Status	Expected Refund
S1	REFUND	INR 128.25
S2	REFUND	INR 135.00
S3	ESCALATE	INR 1,999.00
S4	DENIED	INR 0.00
S5	DENIED	INR 0.00
S6	NOT_FOUND	—
S7	FALLBACK_MAX_ITERATIONS	—

Actual execution details:

Scenario	Actual Status	Refund Amount	Model Calls	Tokens Used
S1	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]
S2	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]
S3	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]
S4	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]
S5	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]	[YOUR RESULT]
S6	[YOUR RESULT]	—	[YOUR RESULT]	[YOUR RESULT]
S7	[YOUR RESULT]	—	[YOUR RESULT]	[YOUR RESULT]

The scenario execution logs were saved to:

outputs/agent_runs.jsonl

8. Safe Execution Limits

Three safety limits were implemented.

Maximum iterations

After six model calls without a final answer, the agent stops with:

FALLBACK_MAX_ITERATIONS

Scenario S7 demonstrates this behaviour.

Token limit

After each model response, token usage is checked. If the token limit is exceeded, the agent stops without executing the requested tool.

S1 was tested with a token limit of 200.

Result: [YOUR TOKEN-LIMIT DEMO STATUS]

Timeout

Elapsed wall-clock time is measured using time.monotonic().

A test using time_limit_seconds=0 stops before the first model call.

Result: [YOUR TIMEOUT DEMO STATUS]

All fallback responses include a short human-handoff summary.

9. Resumable Execution

The agent saves checkpoints after every tool observation.

Each checkpoint contains:

Run ID.
Scenario ID.
Full message history.
Tokens used.
Step number.
Tools already executed.

Checkpoints are stored in:

state/

A simulated crash was introduced immediately after the second tool execution in S1.

The agent was then resumed using resume_agent(run_id).

Tools executed across the crash and resume:

get_order
search_policy
calculate_refund
issue_refund

No tool was repeated.

Final resumed result:

Status: REFUND
Refund amount: INR 128.25
Total tool executions: 4
10. ReAct vs Plan-and-Execute

The ReAct approach makes decisions incrementally based on tool observations. This makes it useful when the agent may encounter unexpected results such as a missing order, a cancelled order or a policy restriction.

A plan-and-execute approach asks the model to generate a complete sequence of actions first, potentially reducing model calls and token usage when the workflow is predictable. However, an unexpected tool result may require replanning and can make a previously generated plan unsuitable.

For this refund workflow, ReAct is preferable because refund decisions depend on actual order status, delivery time and policy retrieval results. Its additional model calls provide more opportunities to validate observations before issuing a refund or escalating a request.

Limitations and Next Steps
Limitations
The supplied dataset is simulated and may not represent real-world Instamart customer behaviour.
TF-IDF retrieval relies on word overlap and does not fully capture semantic meaning.
Customer segmentation depends on the selected RFM features, scaling and clustering parameters.
Lapse predictions are based on historical behaviour and do not guarantee future customer inactivity.
The ticket classifier and refund agent use a mock LLM rather than a production language model.
The Premium flag is based on whole-window RFM data, including the September lapse-label period. Therefore, the retention list is illustrative and not a leakage-free evaluation.
The refund ledger is simulated and does not issue actual refunds.
Next Steps
Introduce semantic embeddings for better policy retrieval.
Evaluate clustering stability across different datasets and time periods.
Improve lapse prediction through additional behavioural features and temporal validation.
Monitor precision and recall as customer behaviour changes.
Add stronger structured-output validation and monitoring for ticket classification.
Integrate real authentication, audit trails and human approval workflows before production use.
Add transactional safeguards and idempotency to refund processing.
Repository Structure
text

instamart/
│
├── data/
│   ├── policies/
│   │   ├── delivery_policy.txt
│   │   ├── escalation_policy.txt
│   │   ├── payment_policy.txt
│   │   ├── privacy_policy.txt
│   │   └── refund_policy.txt
│   │
│   ├── raw/
│   │   ├── agent_scenarios.csv
│   │   ├── customers.csv
│   │   ├── orders.csv
│   │   ├── policy_eval.csv
│   │   ├── products.csv
│   │   └── tickets.csv
│   │
│   └── instamart.db
│
├── outputs/
│   ├── agent_runs.jsonl
│   ├── customer_segments.csv
│   ├── lapse_model.joblib
│   ├── refund_ledger.jsonl
│   ├── retention_list.csv
│   ├── triage_budget_demo.csv
│   └── triage_results.csv
│
├── sql/
│   └── queries.sql
│
├── state/
│   └── checkpoints/
│
├── venv/
│
├── .gitignore
├── audit.py
├── generate.py
├── mock_llm.py
├── model.py
├── support_ticket.py
└── policy_assistant.py



The filenames above are illustrative; keep this tree consistent with the actual repository.

Conclusion

This project demonstrates a complete data-to-decision workflow, beginning with raw data quality checks and ending with AI-assisted customer support automation.

The four parts combine data engineering, SQL analytics, unsupervised learning, supervised machine learning, privacy-aware NLP, retrieval-augmented generation and tool-based agent execution.

The resulting system provides a reproducible foundation for customer segmentation, retention planning, automated ticket triage and policy-grounded refund assistance.