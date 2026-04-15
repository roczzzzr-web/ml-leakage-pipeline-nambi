TASK 1 — Identify Label and Leakage:-
 1.Label :
repeat_purchase_flag
* This is the outcome we are trying to predict (whether the customer repeats a purchase within 30 days).
 2.Data Leakage :
discount_used_on_repeat_order
*This introduces leakage because it contains information only available after the repeat purchase happens, which directly depends on the target.

TASK2 — Missing Steps in ML Workflow:-
Your manager jumped too quickly to modeling. Two critical steps should come first:-

 1.Data Cleaning & Preprocessing:
*This includes handling missing values, checking data types, removing irrelevant columns (like customer_id), and scaling if needed.
Why it matters: Poor data quality leads to unreliable models, no matter how advanced the algorithm is.
 2.Proper Train-Test Split:
*Split the dataset into training and testing sets before any modeling.
Why it matters: This ensures the model is evaluated on unseen data and prevents data leakage, giving a realistic measure of performance.