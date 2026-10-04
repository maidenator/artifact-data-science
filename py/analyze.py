import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
print("Loading data...")
df = pd.read_csv("../dataset/artifacts.csv")

# 2. Define Keeper threshold (90th percentile of final_cv at level 20)
level20_df = df[df['level'] == 20]
threshold = level20_df['final_cv'].quantile(0.90)
df['keeper'] = df['final_cv'] >= threshold
print(f"Keeper threshold (90th percentile final_cv): {threshold:.2f}")

# 3. Descriptive plots (eda.png)
plt.figure(figsize=(12, 5))

# Plot 1: Histogram of final CV
plt.subplot(1, 2, 1)
plt.hist(level20_df['final_cv'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Keeper Threshold ({threshold:.2f})')
plt.title('Histogram of Final CV (+20)')
plt.xlabel('Final CV')
plt.ylabel('Frequency')
plt.legend()

# Plot 2: Mean CV by level
plt.subplot(1, 2, 2)
mean_cv_by_level = df.groupby('level')['cv'].mean()
plt.plot(mean_cv_by_level.index, mean_cv_by_level.values, marker='o', color='purple')
plt.title('Mean CV by Level')
plt.xlabel('Level')
plt.ylabel('Mean CV')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('eda.png')
plt.close()

# 4. Train/Test Split
print("Preparing train/test split...")
unique_ids = df['artifact_id'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.25, random_state=42)

# Create one-hot encoding for categorical features
# Features: cv, sub_count, has_crit_rate, has_crit_dmg, slot, main_stat (one-hot)
df_encoded = pd.get_dummies(df, columns=['slot', 'main_stat'], drop_first=True)
feature_cols = ['cv', 'sub_count', 'has_crit_rate', 'has_crit_dmg'] + \
               [col for col in df_encoded.columns if col.startswith('slot_') or col.startswith('main_stat_')]

# 5. Model training per level
levels = [0, 4, 8, 12, 16]
results = []

print(f"{'Level':<5} | {'Baseline AUC':<12} | {'LR AUC':<10} | {'RF AUC':<10} | {'RF Precision':<12} | {'RF Recall':<10}")
print("-" * 75)

for lvl in levels:
    # Filter by level
    df_lvl = df_encoded[df_encoded['level'] == lvl]
    
    # Split by artifact_id to avoid leakage
    train_df = df_lvl[df_lvl['artifact_id'].isin(train_ids)]
    test_df = df_lvl[df_lvl['artifact_id'].isin(test_ids)]
    
    X_train = train_df[feature_cols]
    y_train = train_df['keeper']
    
    X_test = test_df[feature_cols]
    y_test = test_df['keeper']
    
    # Baseline: ROC-AUC using current cv as the predictor score
    baseline_auc = roc_auc_score(y_test, X_test['cv'])
    
    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_precision = precision_score(y_test, rf_preds)
    rf_recall = recall_score(y_test, rf_preds)
    
    results.append({
        'Level': lvl,
        'Baseline AUC': baseline_auc,
        'LR AUC': lr_auc,
        'RF AUC': rf_auc,
        'RF Precision': rf_precision,
        'RF Recall': rf_recall
    })
    
    print(f"+{lvl:<4} | {baseline_auc:<12.4f} | {lr_auc:<10.4f} | {rf_auc:<10.4f} | {rf_precision:<12.4f} | {rf_recall:<10.4f}")

# 6. Save auc_by_level.png
res_df = pd.DataFrame(results)

plt.figure(figsize=(8, 5))
plt.plot(res_df['Level'], res_df['Baseline AUC'], marker='o', linestyle='--', color='gray', label='Baseline (Current CV)')
plt.plot(res_df['Level'], res_df['LR AUC'], marker='s', color='blue', label='Logistic Regression')
plt.plot(res_df['Level'], res_df['RF AUC'], marker='^', color='green', label='Random Forest')
plt.title('ROC-AUC by Artifact Level')
plt.xlabel('Level')
plt.ylabel('ROC-AUC Score')
plt.xticks(levels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('auc_by_level.png')
plt.close()

print("Saved eda.png and auc_by_level.png")
