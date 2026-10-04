import pandas as pd
df = pd.read_csv("../dataset/artifacts.csv")
threshold = df[df.level == 20].final_cv.quantile(0.90)   # top 10% = keeper
df["keeper"] = df.final_cv >= threshold

print(f"Top 10% final_cv threshold (level 20): {threshold:.2f}")
print(f"Number of keeper artifacts: {df['keeper'].sum()} out of {len(df)}")