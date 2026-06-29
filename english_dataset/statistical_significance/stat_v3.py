import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, binomtest

# Load CSV
df = pd.read_csv("final_results_models.csv")

# ---------------------------
# Helper: best model per group
# ---------------------------
def best_by(df, filters, group_cols):
    sub = df.copy()
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    return sub.groupby(group_cols)["test_macro_f1"].max().reset_index()

# ---------------------------
# H1: LLM vs best classical (Sign test)
# ---------------------------
def h1_llm_vs_classical(df):
    results = []

    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]

        best_classical = sub[sub["model_type"] != "LLM"]["test_macro_f1"].max()
        best_llm = sub[sub["model_type"] == "LLM"]["test_macro_f1"].max()

        results.append(best_llm > best_classical)

    # sign test (binomial)
    n = len(results)
    k = sum(results)

    p_value = binomtest(k, n, 0.5, alternative="greater").pvalue

    return results, p_value


# ---------------------------
# H3/H4/H7: Embedding comparison (Wilcoxon)
# TF-IDF vs contextual embeddings
# ---------------------------
def embedding_test(df, dataset_filter, embedding_a="TF-IDF", embedding_b="DistilBERT"):
    diffs = []

    for dataset in df["dataset"].unique():
        sub = df[(df["dataset"] == dataset) & (df["model_type"] == dataset_filter)]

        a = sub[sub["vectorizer"] == embedding_a]["test_macro_f1"].max()
        b = sub[sub["vectorizer"] == embedding_b]["test_macro_f1"].max()

        diffs.append((a, b))

    a_vals = np.array([x[0] for x in diffs])
    b_vals = np.array([x[1] for x in diffs])

    stat, p = wilcoxon(a_vals, b_vals, alternative="less")

    return a_vals, b_vals, stat, p


# ---------------------------
# H5: LSTM vs BiLSTM (more samples)
# ---------------------------
def h5_lstm_vs_bilstm(df):
    diffs = []

    for _, row in df[df["model_type"] == "DL"].iterrows():
        # pair within same dataset + vectorizer
        sub = df[
            (df["dataset"] == row["dataset"]) &
            (df["vectorizer"] == row["vectorizer"]) &
            (df["model_type"] == "DL")
        ]

        lstm = sub[sub["model"] == "LSTM"]["test_macro_f1"].values
        bilstm = sub[sub["model"] == "BiLSTM"]["test_macro_f1"].values

        if len(lstm) > 0 and len(bilstm) > 0:
            diffs.append((lstm[0], bilstm[0]))

    a = np.array([x[0] for x in diffs])
    b = np.array([x[1] for x in diffs])

    stat, p = wilcoxon(a, b, alternative="less")

    return a, b, stat, p


# ---------------------------
# Run all tests
# ---------------------------
llm_res, h1_p = h1_llm_vs_classical(df)
print("H1 sign test p-value:", h1_p)

a, b, stat_h3, p_h3 = embedding_test(df, "ML", "TF-IDF", "DistilBERT")
print("H3 Wilcoxon p-value:", p_h3)

a, b, stat_h4, p_h4 = embedding_test(df, "ML", "TF-IDF", "CamemBERT")
print("H4 Wilcoxon p-value:", p_h4)

a, b, stat_h7, p_h7 = embedding_test(df, "ML", "TF-IDF", "AraBERT")
print("H7 Wilcoxon p-value:", p_h7)

a, b, stat_h5, p_h5 = h5_lstm_vs_bilstm(df)
print("H5 Wilcoxon p-value:", p_h5)