import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# 1. LOAD THE DATA
# Make sure this points to the file your EEG pipeline just generated
df = pd.read_csv(r"C:\Users\Neermita\Desktop\memory_and_task\code\results\ers_metrics.csv")

# Drop any sessions that might have failed and returned NaN
df = df.dropna(subset=["acc", "correct_recall_rate"])

# 2. RUN THE LINEAR MIXED-EFFECTS MODEL
# Tests if ERS accuracy predicts memory, controlling for subject differences
print("--- Linear Mixed-Effects Model Results ---")
model = smf.mixedlm("correct_recall_rate ~ acc", df, groups=df["subject"])
result = model.fit()
print(result.summary())

# Look at the P>|z| column for 'acc'. If it's < 0.05, ERS predicts memory!

# 3. PLOT THE RESULTS
plt.figure(figsize=(9, 6))

# Scatter plot of individual sessions, colored by patient
sns.scatterplot(
    data=df, 
    x="acc", 
    y="correct_recall_rate", 
    hue="subject", 
    palette="Set2", 
    s=70, 
    alpha=0.8
)

# Draw the overall fixed-effect trendline from the mixed model
b0 = result.params["Intercept"]
b1 = result.params["acc"]
x_vals = df["acc"]
y_vals = b0 + b1 * x_vals

plt.plot(x_vals, y_vals, color="black", linewidth=2.5, label="Overall Trend (Fixed Effect)")

# Formatting for publication
plt.title("ERS Decoding Accuracy Predicts Episodic Memory", fontsize=14, fontweight="bold")
plt.xlabel("Encoding-Retrieval State (ERS) Decoding Accuracy", fontsize=12)
plt.ylabel("Correct Recall Rate", fontsize=12)
plt.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance Level (0.50)")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig(r"C:\Users\Neermita\Desktop\memory_and_task\code\results\ers_vs_behavior.png", dpi=300)
plt.show()