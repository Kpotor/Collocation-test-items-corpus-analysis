import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. INITIALIZATION AND CONFIGURATION
# =============================================================================

# Color palette settings
test_colors = {
    'GFS2015': '#1f77b4',
    'NW2017': '#ff7f0e',
    'COLLEX': '#2ca02c',
    'COLLMATCH': '#d62728',
    'CONTRIX': '#9467bd',
    'ACT': '#072c62'
}

# =============================================================================
# 2. DATA LOADING AND PREPROCESSING
# =============================================================================

# Data loading
df = pd.read_csv('item_list_with_statistics_v2.csv')

# Extract valid data
df_valid = df[df['MI'].notna()].copy()

# Column name definitions
genre_cols = ['acad', 'blog', 'fic', 'mag', 'news', 'spok', 'tvm', 'web']
stat_cols = ['MI', 't-score', 'z-score', 'logDice', 'ΔP(w1->w2)', 'ΔP(w2->w1)']

# Remove rows from df_valid where idx column values are in the specified list
values_to_remove = [61, 120, 139] 
df_valid = df_valid[~df_valid['idx'].isin(values_to_remove)].copy()
df_valid = df_valid[(df_valid["rel_type"] == "mod_noun") | (df_valid["rel_type"] == "verb_obj")]


# Modify source element names
source_mapping = {
    'Gonzalez_Fernandez_and_Schmitt_2015': 'GFS2015',
    'Nguyen_and_Webb_2017': 'NW2017',
    'Gyllstad_2007_COLLEX': 'COLLEX',
    'Gyllstad_2007_COLLMATCH': 'COLLMATCH',
    'Reiver_2014_CONTRIX': 'CONTRIX',
    'Nyuyen_2022_ACT': 'ACT'
}

df_valid.to_csv("Test_items_targeted.csv", encoding="utf-8", index=False)

# Calculate log frequency
df_valid["log_freq"] = np.log10(df_valid["total_freq"])

# Modify source element names
source_mapping = {
    'Gonzalez_Fernandez_and_Schmitt_2015': 'GFS2015',
    'Nguyen_and_Webb_2017': 'NW2017',
    'Gyllstad_2007_COLLEX': 'COLLEX',
    'Gyllstad_2007_COLLMATCH': 'COLLMATCH',
    'Reiver_2014_CONTRIX': 'CONTRIX',
    'Nyuyen_2022_ACT': 'ACT'
}

df_valid['source'] = df_valid['source'].map(source_mapping)


print(f"Total items: {len(df)}")
print(f"Valid items: {len(df_valid)}")
print(f"\nItems by test:")
print(df_valid['source'].value_counts())

# =============================================================================
# 3. VISUALIZATION: DISTRIBUTION PLOTS
# =============================================================================

# 3.1 Log Frequency Distribution Plot
# Draw raincloud plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
order = list(test_colors.keys())
palette = [test_colors[src] for src in order]

# Violin plot (distribution)
sns.violinplot(
    data=df_valid,
    x="source",
    y="log_freq",
    order=order,
    palette=palette,
    inner=None,
    cut=0,
    linewidth=1
)

# Make strip plot (individual data points) more visible with darker color (black)
sns.stripplot(
    data=df_valid,
    x="source",
    y="log_freq",
    order=order,
    color="black",  # Display data points in black
    alpha=0.3,      # Make almost opaque for better visibility
    jitter=True,
    size=4
)

# Add mean and median to plot
import numpy as np

for i, src in enumerate(order):
    subset = df_valid[df_valid["source"] == src]["log_freq"]
    mean = subset.mean()
    median = subset.median()
    # Mean (red line)
    plt.plot([i-0.25, i+0.25], [mean, mean], color="red", lw=2, label="mean" if i==0 else "")
    # Median (blue line)
    plt.plot([i-0.25, i+0.25], [median, median], color="blue", lw=2, linestyle="--", label="median" if i==0 else "")

# Display legend only once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper right")

plt.ylabel("log frequency")
plt.title("Log-frequency distribution for each source", fontsize=20)
plt.tight_layout()
plt.show()

# 3.2 MI Distribution Plot
# Draw raincloud plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
order = list(test_colors.keys())
palette = [test_colors[src] for src in order]

# Violin plot (distribution)
sns.violinplot(
    data=df_valid,
    x="source",
    y="MI",
    order=order,
    palette=palette,
    inner=None,
    cut=0,
    linewidth=1
)

# Make strip plot (individual data points) more visible with darker color (black)
sns.stripplot(
    data=df_valid,
    x="source",
    y="MI",
    order=order,
    color="black",  # Display data points in black
    alpha=0.3,      # Make almost opaque for better visibility
    jitter=True,
    size=4
)

# Add mean and median to plot
import numpy as np

for i, src in enumerate(order):
    subset = df_valid[df_valid["source"] == src]["MI"]
    mean = subset.mean()
    median = subset.median()
    # Mean (red line)
    plt.plot([i-0.25, i+0.25], [mean, mean], color="red", lw=2, label="mean" if i==0 else "")
    # Median (blue line)
    plt.plot([i-0.25, i+0.25], [median, median], color="blue", lw=2, linestyle="--", label="median" if i==0 else "")

# Display legend only once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper right")

plt.ylabel("MI")
plt.title("MI distribution for each source", fontsize=20)
plt.tight_layout()
plt.show()

# 3.3 T-score Distribution Plot
# Draw raincloud plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
order = list(test_colors.keys())
palette = [test_colors[src] for src in order]

# Violin plot (distribution)
sns.violinplot(
    data=df_valid,
    x="source",
    y="t-score",
    order=order,
    palette=palette,
    inner=None,
    cut=0,
    linewidth=1
)

# Make strip plot (individual data points) more visible with darker color (black)
sns.stripplot(
    data=df_valid,
    x="source",
    y="t-score",
    order=order,
    color="black",  # Display data points in black
    alpha=0.3,      # Make almost opaque for better visibility
    jitter=True,
    size=4
)

# Add mean and median to plot
import numpy as np

for i, src in enumerate(order):
    subset = df_valid[df_valid["source"] == src]["t-score"]
    mean = subset.mean()
    median = subset.median()
    # Mean (red line)
    plt.plot([i-0.25, i+0.25], [mean, mean], color="red", lw=2, label="mean" if i==0 else "")
    # Median (blue line)
    plt.plot([i-0.25, i+0.25], [median, median], color="blue", lw=2, linestyle="--", label="median" if i==0 else "")

# Display legend only once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper right")

plt.ylabel("t-score")
plt.title("T-score distribution for each source", fontsize=20)
plt.tight_layout()
plt.show()

# 3.4 LogDice Distribution Plot
# Draw raincloud plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
order = list(test_colors.keys())
palette = [test_colors[src] for src in order]

# Violin plot (distribution)
sns.violinplot(
    data=df_valid,
    x="source",
    y="logDice",
    order=order,
    palette=palette,
    inner=None,
    cut=0,
    linewidth=1
)

# Make strip plot (individual data points) more visible with darker color (black)
sns.stripplot(
    data=df_valid,
    x="source",
    y="logDice",
    order=order,
    color="black",  # Display data points in black
    alpha=0.3,      # Make almost opaque for better visibility
    jitter=True,
    size=4
)

# Add mean and median to plot
import numpy as np

for i, src in enumerate(order):
    subset = df_valid[df_valid["source"] == src]["logDice"]
    mean = subset.mean()
    median = subset.median()
    # Mean (red line)
    plt.plot([i-0.25, i+0.25], [mean, mean], color="red", lw=2, label="mean" if i==0 else "")
    # Median (blue line)
    plt.plot([i-0.25, i+0.25], [median, median], color="blue", lw=2, linestyle="--", label="median" if i==0 else "")

# Display legend only once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper right")

plt.ylabel("logDice")
plt.title("LogDice distribution for each source", fontsize=20)
plt.tight_layout()
plt.show()

# =============================================================================
# 4. STATISTICAL ANALYSIS: ASSOCIATION MEASURES
# =============================================================================

target_cols = ['log_freq', 'MI', 't-score', 'z-score', 'logDice', 'ΔP(w1->w2)', 'ΔP(w2->w1)']

X = df_valid[target_cols]

# Winsorization for strength measures (1-99 percentile)
for c in ["MI","t-score","z-score","logDice", 'ΔP(w1->w2)', 'ΔP(w2->w1)']:
    lo, hi = X[c].quantile([0.01, 0.90])
    X[c] = X[c].clip(lo, hi)

ams_grouped_mean = (
    df_valid.groupby("source", dropna=False)[target_cols]
      .mean(numeric_only=True)
      .sort_index()
)


ams_grouped_mean_z = ams_grouped_mean.apply(stats.zscore, axis=0)

# 4.1 Clustermap of Association Measures
# ---- matrix to plot (row-centered values you already computed) ----
mat = ams_grouped_mean_z.loc[:, target_cols].copy()
mat = mat.replace([np.inf, -np.inf], np.nan).dropna(how="all")

# ---- clustermap ----
# sns.set(style="white", context="notebook", font="DejaVu Sans")

g = sns.clustermap(
    mat,
    metric="euclidean",       # distance for clustering
    method="ward",         # linkage method
    annot=True,
    cmap="coolwarm",              # diverging colormap
    center=0.0,               # 0 is the row mean
    linewidths=0.5,
    figsize=(12, 8),
    row_cluster=True,
    col_cluster=False,
)

# titles & labels in English
g.fig.suptitle("Cluster Map of Test Items Statistics by Source", y=1.02, fontsize=20)
g.ax_heatmap.set_xlabel("AMs", fontsize=12)
g.ax_heatmap.set_ylabel("Source", fontsize=12)

# improve tick readability and set tick label size
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=14)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=14)

plt.tight_layout()
plt.show()

# Optionally save
# g.savefig("grouped_mean_centered_clustermap.png", dpi=300, bbox_inches="tight")

# =============================================================================
# 5. GENRE ANALYSIS
# =============================================================================

# Analyze genre bias for each test
for col in genre_cols:
    df_valid[f"{col}_log"] = np.log10(df_valid[col] + 1)

genre_cols_log = [f"{col}_log" for col in genre_cols]

grouped_mean = (
    df_valid.groupby("source", dropna=False)[genre_cols_log]
      .mean(numeric_only=True)
      .sort_index()
)

grouped_mean["row_mean"] = grouped_mean.mean(axis=1)
grouped_mean_centered = grouped_mean[genre_cols_log].sub(grouped_mean["row_mean"], axis=0)

# 5.1 Clustermap of Genre Frequencies
# ---- matrix to plot (row-centered values you already computed) ----
mat = grouped_mean_centered.loc[:, genre_cols_log].copy()
mat = mat.replace([np.inf, -np.inf], np.nan).dropna(how="all")

# ---- clustermap ----
# sns.set(style="white", context="notebook", font="DejaVu Sans")

g = sns.clustermap(
    mat,
    metric="euclidean",       # distance for clustering
    method="ward",         # linkage method
    annot=True,
    cmap="coolwarm",              # diverging colormap
    center=0.0,               # 0 is the row mean
    linewidths=0.5,
    figsize=(8,8),
    # cbar_kws={"label": "Row-centered log10 frequency"},  # colorbar label in English
)

# titles & labels in English
g.fig.suptitle("Cluster Map of Row-Centered Genre Frequencies by Source", y=1.02, fontsize=20)
g.ax_heatmap.set_xlabel("Genres (log10)", fontsize=12)
g.ax_heatmap.set_ylabel("Source", fontsize=12)

# improve tick readability
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=14)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=14)

plt.tight_layout()
plt.show()

# Optionally save
# g.savefig("grouped_mean_centered_clustermap.png", dpi=300, bbox_inches="tight")

# =============================================================================
# 6. PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================

preprocessed_df = df_valid.copy()
# Winsorization for strength measures (1-99 percentile)
for c in ["MI","t-score","z-score","logDice", 'ΔP(w1->w2)', 'ΔP(w2->w1)']:
    lo, hi = preprocessed_df[c].quantile([0.01, 0.99])
    preprocessed_df[c] = preprocessed_df[c].clip(lo, hi)

# 6.1 Data Preparation for PCA
cols = ["log_freq","MI","t-score","z-score","logDice","ΔP(w1->w2)","ΔP(w2->w1)"]
X = preprocessed_df[cols].copy()

# 6.2 PCA Implementation
# 1) Preprocessing
# Standardization
Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=cols, index=X.index)

# 2) PCA
pca = PCA()  # Number of components to be determined later
scores = pca.fit_transform(Z)
explained = pca.explained_variance_ratio_
loadings = pd.DataFrame(pca.components_.T, index=cols,
                        columns=[f"PC{i+1}" for i in range(len(cols))])


# Add PC1-PC3 scores to original dataframe
pc_scores = pd.DataFrame(scores[:, :3], 
                        columns=['PC1', 'PC2', 'PC3'], 
                        index=preprocessed_df.index)

# Combine PC1-PC3 scores with preprocessed_df
preprocessed_df = pd.concat([preprocessed_df, pc_scores], axis=1)

# 6.3 Save Results
preprocessed_df.to_csv("items_statistics_with_PCA.csv")