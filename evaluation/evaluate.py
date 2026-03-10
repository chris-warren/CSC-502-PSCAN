import pandas as pd

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

# Load predictions and ground truth data
gt_df = pd.read_csv("data/labels/lfr_5000.labels.tsv", sep="\t").rename(
    columns={"label": "gt_label"}
)
pred_df = pd.read_csv("data/labels/lfr_5000.labels.tsv", sep="\t").rename(
    columns={"label": "predicted_label"}
)
df = pd.merge(
    gt_df,
    pred_df,
    on="node",
).reset_index()
print(df.head())

# compute ARI
ari = adjusted_rand_score(df["gt_label"], df["predicted_label"])
print(f"adjusted rand index score: {ari}")

# compute NMI
nmi = normalized_mutual_info_score(df["gt_label"], df["predicted_label"])
print(f"normalized mutual info score: {nmi}")
