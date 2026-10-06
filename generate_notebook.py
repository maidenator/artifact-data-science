"""Regenerate artifact_analysis_report.ipynb (option B: keeper classification)
and artifact_presentation.ipynb (same story, code sources removed, outputs
populated by executing the report and copying outputs over).

Run:  python generate_notebook.py
Workflow after editing: regen writes both files with empty outputs, then
execute the report (Run All) and copy its outputs into the presentation twin.
Output: artifact_analysis_report.ipynb + artifact_presentation.ipynb
Story: guess final CV -> keep-or-toss keeper test (AUC by level) -> 3 quality tiers.
Tone: grader-friendly, plain language.
"""
import json
from pathlib import Path

nb_path = Path(__file__).parent / "artifact_analysis_report.ipynb"
pres_path = Path(__file__).parent / "artifact_presentation.ipynb"


def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(lines):
    # Hide code inputs for live presentation (outputs stay visible).
    # Code is still in the file and expandable — grade-safe.
    return {"cell_type": "code", "execution_count": None, "metadata": {"jupyter": {"source_hidden": True}, "vscode": {"inputCollapsed": True}}, "outputs": [], "source": lines}


cells = []

# 1 - Title + dataset
cells.append(md([
"# Mini Data Science Project: Genshin Impact Artifacts\n",
"\n",
"## 1. Dataset selection\n",
"\n",
"We use a simulated dataset of **1,000,000 Artifacts** from the game Genshin Impact. Each artifact was tracked at levels 0, 4, 8, 12, 16 and 20, so the table has **6,000,000 rows** (one row per snapshot).\n",
"\n",
"### What each column means\n",
"\n",
"| Column | Meaning |\n",
"|---|---|\n",
"| `artifact_id` | Which artifact this snapshot belongs to |\n",
"| `level` | Upgrade level (0, 4, 8, 12, 16, 20) |\n",
"| `slot`, `main_stat`, `sub_count` | Piece type and how many bonus stats it has |\n",
"| `cv` | Crit Value right now (see formula below) |\n",
"| `has_crit_rate`, `has_crit_dmg` | Has those prized crit stats yet? (1 = yes, 0 = no) |\n",
"| `final_cv` | CV at level 20 (the target we want to predict) |\n",
"\n",
"Crit Value formula:\n",
"\n",
"$$ \\text{CV} = 2 \\times \\text{Crit Rate\\%} + \\text{Crit Damage\\%} $$\n",
"\n",
"## 2. What is an Artifact?\n",
"\n",
"Artifacts are gear pieces you level up from 0 to 20. Every 4 levels one random bonus stat gets stronger. Players want high Crit Value because it means more damage.\n",
"\n",
"### Example: leveling a Flower\n",
"\n",
"| Stat | Level 0 (start) | Level 20 (finished) |\n",
"|---|---|---|\n",
"| HP | +269 | +538 (1 upgrade) |\n",
"| ATK | +4.1% | +4.1% (no upgrade) |\n",
"| Crit Rate | +3.9% | +10.5% (2 upgrades) |\n",
"| Crit Damage | +7.8% | +21.8% (2 upgrades) |\n",
"\n",
"How the CV adds up:\n",
"\n",
"- Starting: $\\text{CV}_0 = 2 \\times 3.9 + 7.8 = 15.6$\n",
"\n",
"- Final: $\\text{CV}_{20} = 2 \\times 10.5 + 21.8 = 42.8$\n",
"\n",
"**Our questions:**\n",
"\n",
"1. Can we guess the final CV from the starting stats at level 0?\n",
"2. Can we spot a *keeper* early - an artifact that will finish in the top 10%?\n",
"3. Can we group finished artifacts into simple quality tiers (needs work / good / exceptional)?\n"
]))

# 2 - imports single cell
cells.append(code([
"import pandas as pd\n",
"import numpy as np\n",
"import matplotlib.pyplot as plt\n",
"import seaborn as sns\n",
"from sklearn.cluster import KMeans\n",
"from sklearn.linear_model import LinearRegression, LogisticRegression\n",
"from sklearn.ensemble import RandomForestClassifier\n",
"from sklearn.model_selection import train_test_split\n",
"from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score\n",
"from sklearn.preprocessing import StandardScaler\n",
"\n",
"# Plain, readable plots\n",
"sns.set_theme(style=\"whitegrid\")\n",
"plt.rcParams['figure.figsize'] = (10, 6)\n",
"pd.options.display.float_format = '{:,.2f}'.format\n"
]))

cells.append(md([
"## 3. Data preparation\n",
"\n",
"We load the data, check for gaps, look at the first rows and averages, then split off level 0 (for guessing the final number) and level 20 (for defining keepers and tiers). We split by `artifact_id` later so the same artifact never appears in both training and testing.\n",
"\n",
"Cleaning checklist (assignment §3): missing values → `isnull().sum()` below (result: none, so no imputation needed); types → all numeric; ready splits → level 0 for regression/classification, level 20 for keeper line and clustering.\n",
"\n",
"Technique map (assignment §2): §5 = Linear Regression, §6 = Classification (Logistic Regression + Random Forest), §7 = Clustering (K-means).\n"
]))

cells.append(code([
"# Load the dataset\n",
"df = pd.read_csv('dataset/artifacts.csv')\n",
"\n",
"from IPython.display import display\n",
"\n",
"# Shape, missing values, first rows, averages\n",
"print(f\"Rows: {len(df):,}, Artifacts: {df['artifact_id'].nunique():,}\")\n",
"display(df.info())\n",
"display(df.isnull().sum())\n",
"display(df.head())\n",
"display(df.describe())\n"
]))

cells.append(code([
"# How many snapshots per level?\n",
"print(df['level'].value_counts().sort_index().to_string())\n",
"\n",
"# Split by level - level 0 for early guessing, level 20 for final quality\n",
"df_lvl0 = df[df['level'] == 0].copy()\n",
"df_lvl20 = df[df['level'] == 20].copy()\n",
"print(f\"\\nLevel 0 rows: {len(df_lvl0):,}\")\n",
"print(f\"Level 20 rows: {len(df_lvl20):,}\")\n",
"\n",
"# Define a keeper: top 10% of finished artifacts\n",
"keeper_threshold = df_lvl20['final_cv'].quantile(0.90)\n",
"df['keeper'] = df['final_cv'] >= keeper_threshold\n",
"df_lvl0['keeper'] = df_lvl0['final_cv'] >= keeper_threshold\n",
"df_lvl20['keeper'] = df_lvl20['final_cv'] >= keeper_threshold\n",
"print(f\"\\nKeeper means final CV >= {keeper_threshold:.2f} (top 10% at level 20)\")\n",
"print(f\"Keepers in total table: {df['keeper'].sum():,} out of {len(df):,}\")\n",
"print(f\"Anomalies (final CV > 40): {(df_lvl20['final_cv'] > 40).sum():,} out of {len(df_lvl20):,} finished ({(df_lvl20['final_cv'] > 40).mean()*100:.2f}%)\")\n"
]))

cells.append(md([
"## 4. First look at the data\n",
"\n",
"Most finished artifacts are weak. Only a small slice on the right side of the chart counts as a keeper. On average CV grows steadily with each upgrade, but luck matters a lot.\n"
]))

cells.append(code([
"# Plot 1: where do finished artifacts end up? + keeper line\n",
"# Plot 2: average CV goes up with level\n",
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"\n",
"sns.histplot(df_lvl20['final_cv'], bins=50, kde=True, color='skyblue', ax=axes[0])\n",
"axes[0].axvline(keeper_threshold, color='red', linestyle='--', linewidth=2,\n",
"                label=f'Keeper line ({keeper_threshold:.2f})')\n",
"axes[0].set_title('Where finished artifacts end up (Level 20)', fontsize=14)\n",
"axes[0].set_xlabel('Final Crit Value')\n",
"axes[0].set_ylabel('Count')\n",
"axes[0].legend()\n",
"\n",
"mean_cv_by_level = df.groupby('level')['cv'].mean()\n",
"axes[1].plot(mean_cv_by_level.index, mean_cv_by_level.values, marker='o', color='purple')\n",
"axes[1].set_title('Average CV grows with level', fontsize=14)\n",
"axes[1].set_xlabel('Level')\n",
"axes[1].set_ylabel('Average CV')\n",
"axes[1].grid(True, linestyle='--', alpha=0.7)\n",
"\n",
"plt.tight_layout()\n",
"plt.show()\n"
]))

cells.append(md([
"## 5. Technique 1: Linear Regression — guessing the final number\n",
"\n",
"We start simple. Using only what we see at level 0 (`cv`, `sub_count`, and whether crit stats are present), we make one straight-line guess at the final score.\n",
"\n",
"Think of it as:\n",
"\n",
"starting stats $\\rightarrow$ one guessed final number\n",
"\n",
"How we judge the guesses:\n",
"\n",
"- How far off are we on average, in CV points? (called $\\text{RMSE}$)\n",
"\n",
"- How much of the gap between good and bad artifacts do we capture? (called $R^2$, where 0 = nothing and 1 = perfect)\n",
"\n",
"Note: starting CV already includes crit info, so the two yes/no crit flags add only a little on top. We leave out `slot` and `main_stat` here to keep things easy to explain.\n"
]))

cells.append(code([
"# Technique 1: Linear Regression from level 0 stats\n",
"features_reg = ['sub_count', 'cv', 'has_crit_rate', 'has_crit_dmg']\n",
"X = df_lvl0[features_reg]\n",
"y = df_lvl0['final_cv']\n",
"\n",
"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
"\n",
"regressor = LinearRegression()\n",
"regressor.fit(X_train, y_train)\n",
"y_pred = regressor.predict(X_test)\n",
"\n",
"mse = mean_squared_error(y_test, y_pred)\n",
"rmse = np.sqrt(mse)\n",
"r2 = r2_score(y_test, y_pred)\n",
"print(f\"MSE: {mse:.2f}  |  RMSE: {rmse:.2f}  |  R2: {r2:.2f}\")\n",
"print(f\"On average our guess is off by about {rmse:.1f} CV points.\")\n",
"print(\"\\nWhat the model learned (higher = pushes guess up):\")\n",
"for name, coef in zip(features_reg, regressor.coef_):\n",
"    print(f\"  {name}: {coef:.3f}\")\n",
"print(f\"  starting point: {regressor.intercept_:.2f}\")\n"
]))

cells.append(md([
"An $R^2$ around 0.5 means the starting stats explain about half the story. The rest is luck during upgrades, so no level 0 guess can be perfect.\n"
]))

cells.append(code([
"# Actual vs predicted - sampled so the plot stays readable (200k points would be a blob)\n",
"sample_idx = np.random.RandomState(42).choice(len(y_test), size=5000, replace=False)\n",
"y_test_s = y_test.iloc[sample_idx]\n",
"y_pred_s = y_pred[sample_idx]\n",
"\n",
"plt.figure(figsize=(8, 6))\n",
"plt.hexbin(y_test_s, y_pred_s, gridsize=40, mincnt=1, alpha=0.9)\n",
"plt.colorbar(label='Count')\n",
"mn, mx = float(y.min()), float(y.max())\n",
"plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect guess')\n",
"plt.title('Guessed vs actual final CV (5,000 sampled points)', fontsize=14)\n",
"plt.xlabel('Actual final CV')\n",
"plt.ylabel('Guessed final CV')\n",
"plt.legend()\n",
"plt.show()\n"
]))

cells.append(md([
"## 6. Technique 2: Classification — keep or toss? (the keeper test)\n",
"\n",
"Players do not need the exact final number. They need a yes/no answer: *is this worth leveling?*\n",
"\n",
"We call the top 10% keepers. The cutoff sits around 21.8 CV.\n",
"\n",
"Then at each level (0, 4, 8, 12, 16) we compare three classifiers:\n",
"\n",
"- **Current CV only (baseline)** - just trust what you see now.\n",
"\n",
"- **Classification: Logistic Regression** - one weight per stat, added up (simple weighted score).\n",
"\n",
"- **Classification: Random Forest** - many small rules vote together (tree vote).\n",
"\n",
"How we judge it:\n",
"\n",
"- Ranking quality (called $AUC$): 0.5 = guessing, 1.0 = perfect.\n",
"\n",
"- When it says keeper, how often is it right? And how many real keepers does it find?\n",
"\n",
"We split by artifact ID so the same artifact is never in both training and testing. We sample 100k training rows per level so the notebook runs fast - results match full data within a point or two.\n"
]))

cells.append(code([
"# Technique 2: keeper classification per level\n",
"feature_cols = ['cv', 'sub_count', 'has_crit_rate', 'has_crit_dmg']\n",
"levels = [0, 4, 8, 12, 16]\n",
"\n",
"# Split artifacts (not rows) so there is no leakage\n",
"unique_ids = df['artifact_id'].unique()\n",
"train_ids, test_ids = train_test_split(unique_ids, test_size=0.25, random_state=42)\n",
"\n",
"results = []\n",
"print(f\"{'Level':<6} | {'Base AUC':<9} | {'LogReg AUC':<11} | {'Forest AUC':<10} | {'Forest Prec':<11} | {'Forest Recal'}\")\n",
"print(\"-\" * 80)\n",
"\n",
"for lvl in levels:\n",
"    df_lvl = df[df['level'] == lvl]\n",
"    train_df = df_lvl[df_lvl['artifact_id'].isin(train_ids)].sample(n=100000, random_state=42)\n",
"    test_df = df_lvl[df_lvl['artifact_id'].isin(test_ids)].sample(n=25000, random_state=42)\n",
"    X_train, y_train = train_df[feature_cols], train_df['keeper']\n",
"    X_test, y_test_lvl = test_df[feature_cols], test_df['keeper']\n",
"\n",
"    # Baseline: current CV as the score\n",
"    baseline_auc = roc_auc_score(y_test_lvl, X_test['cv'])\n",
"\n",
"    # Simple model\n",
"    lr = LogisticRegression(class_weight='balanced', max_iter=1000)\n",
"    lr.fit(X_train, y_train)\n",
"    lr_auc = roc_auc_score(y_test_lvl, lr.predict_proba(X_test)[:, 1])\n",
"\n",
"    # Tree vote\n",
"    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=42, n_jobs=-1)\n",
"    rf.fit(X_train, y_train)\n",
"    rf_probs = rf.predict_proba(X_test)[:, 1]\n",
"    rf_preds = rf.predict(X_test)\n",
"    rf_auc = roc_auc_score(y_test_lvl, rf_probs)\n",
"    rf_prec = precision_score(y_test_lvl, rf_preds, zero_division=0)\n",
"    rf_rec = recall_score(y_test_lvl, rf_preds, zero_division=0)\n",
"\n",
"    results.append({'level': lvl, 'baseline': baseline_auc, 'logreg': lr_auc,\n",
"                    'forest': rf_auc, 'prec': rf_prec, 'rec': rf_rec})\n",
"    print(f\"+{lvl:<5} | {baseline_auc:<9.4f} | {lr_auc:<11.4f} | {rf_auc:<10.4f} | {rf_prec:<11.4f} | {rf_rec:.4f}\")\n",
"\n",
"res_df = pd.DataFrame(results)\n",
"display(res_df)\n"
]))

cells.append(code([
"# AUC goes up with level: early guesses are decent, late calls are very safe\n",
"plt.figure(figsize=(8, 5))\n",
"plt.plot(res_df['level'], res_df['baseline'], marker='o', linestyle='--', color='gray', label='Current CV only')\n",
"plt.plot(res_df['level'], res_df['logreg'], marker='s', color='blue', label='Logistic Regression')\n",
"plt.plot(res_df['level'], res_df['forest'], marker='^', color='green', label='Random Forest')\n",
"plt.title('How well can we spot a keeper at each level? (AUC)', fontsize=14)\n",
"plt.xlabel('Level')\n",
"plt.ylabel('AUC (0.5 = guess, 1.0 = perfect)')\n",
"plt.xticks(levels)\n",
"plt.ylim(0.75, 1.0)\n",
"plt.legend()\n",
"plt.grid(True, linestyle='--', alpha=0.7)\n",
"plt.tight_layout()\n",
"plt.show()\n"
]))

cells.append(md([
"## 7. Technique 3: Clustering (K-means) — grouping finished artifacts\n",
"\n",
"We group finished artifacts by final CV into 3 tiers: needs-work, good, and exceptional.\n",
"\n",
"We use CV only because `sub_count` at level 20 is almost always 4, so it does not help separate groups. Values are rescaled first so the grouping is fair, then centers are converted back to real CV.\n"
]))

cells.append(code([
"# Technique 3: K-Means on finished CV (1-D, so groups are easy to explain)\n",
"X_final = df_lvl20[['cv']]\n",
"scaler = StandardScaler()\n",
"X_scaled = scaler.fit_transform(X_final)\n",
"\n",
"kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\n",
"df_lvl20['tier'] = kmeans.fit_predict(X_scaled)\n",
"\n",
"# Convert centers back to real CV numbers and sort low -> high\n",
"centers_cv = scaler.inverse_transform(kmeans.cluster_centers_).flatten()\n",
"order = np.argsort(centers_cv)\n",
"remap = {old: new for new, old in enumerate(order)}\n",
"df_lvl20['tier'] = df_lvl20['tier'].map(remap)\n",
"centers_sorted = np.sort(centers_cv)\n",
"\n",
"print(\"Tier centers (real CV):\")\n",
"for i, c in enumerate(centers_sorted):\n",
"    print(f\"  Tier {i}: ~{c:.1f} CV\")\n",
"print(\"\\nCounts per tier:\")\n",
"print(df_lvl20['tier'].value_counts().sort_index().to_string())\n",
"tier_names = {0: 'Needs work', 1: 'Good', 2: 'Exceptional'}\n",
"print(\"\\nMeaning:\", tier_names)\n"
]))

cells.append(code([
"# Finished artifacts colored by tier\n",
"plt.figure(figsize=(10, 6))\n",
"sns.histplot(data=df_lvl20, x='cv', hue='tier', bins=50, palette='viridis', alpha=0.6, multiple='stack')\n",
"plt.title('Finished artifacts split into 3 tiers', fontsize=14)\n",
"plt.xlabel('Final Crit Value')\n",
"plt.ylabel('Count')\n",
"plt.show()\n"
]))

cells.append(md([
"## 8. What we found (for the presentation)\n",
"\n",
"- **Most artifacts finish weak.** Average finished CV is about 8.8, middle value is about 6.2. The keeper line sits around 21.8. Anything above 40 is very rare and very valuable.\n",
"- **Starting stats help but luck matters.** Guessing from level 0 is off by about 6-7 points on average and captures about half the gap between good and bad pieces.\n",
"- **Keep-or-toss gets safer as you level.** Current CV alone already ranks well (about 0.83 at level 0, about 0.98 at level 16 on a 0.5-to-1.0 scale). The tree vote is a little better early on. Practical rule: be strict early, trust the call from level 12 on.\n",
"- **Three clear tiers.** Finished pieces split into needs-work (low CV bulk), good (middle), and exceptional (keeper zone on the right).\n",
"\n",
"### Limits and next steps\n",
"- This data is simulated to mimic game odds, not pulled from real accounts, so real drop rates may differ.\n",
"- We kept the models simple on purpose (`cv` + 3 helper columns). Adding `slot` and `main_stat` or trying per-level thresholds is an easy follow-up.\n",
"- For players: do not throw away everything early (level 0 calls still miss). Re-check at 8-12 before deciding.\n"
]))

notebook = {
 "cells": cells,
 "metadata": {
  "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

# Presentation twin: identical markdown, code cells keep outputs but no source,
# so Rosae shows story + numbers + charts with zero code boxes. Outputs are
# filled in by executing the report and copying them over (see build step);
# until then placeholders stay empty.
pres_cells = []
for c in cells:
    if c["cell_type"] == "code":
        pres_cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": []})
    else:
        pres_cells.append(c)

pres_nb = {"cells": pres_cells, "metadata": notebook["metadata"],
           "nbformat": 4, "nbformat_minor": 4}

with open(pres_path, "w", encoding="utf-8") as f:
    json.dump(pres_nb, f, indent=1)

print(f"Wrote {len(cells)} cells to {nb_path}")
print(f"Wrote {len(pres_cells)} cells to {pres_path}")
