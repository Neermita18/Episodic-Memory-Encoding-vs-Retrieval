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

## Create the figure
plt.figure(figsize=(10, 7))

# 1. Use a high-contrast, publication-ready color palette
high_contrast_palette = sns.color_palette("Dark2", 5)

# Plot the scatter points
ax = sns.scatterplot(
    data=df, 
    x="acc", 
    y="correct_recall_rate", 
    hue="subject", 
    palette=high_contrast_palette, 
    s=200,             # Larger dots
    alpha=0.99,         # Less transparent for stronger color
    edgecolor="white", # Crisp white borders around dots so they don't blend
    linewidth=1
)

# Draw the overall fixed-effect trendline
b0 = result.params["Intercept"]
b1 = result.params["acc"]
x_vals = df["acc"]
y_vals = b0 + b1 * x_vals

# Thicker, bolder trendline
plt.plot(x_vals, y_vals, color="black", linewidth=3.5, alpha=0.8)
plt.axvline(0.5, color="gray", linestyle="--", linewidth=2, alpha=0.8)

# 2. Increase label font sizes significantly
plt.xlabel("Encoding-Retrieval State Decoding Accuracy", fontsize=16, fontweight="bold", labelpad=10)
plt.ylabel("Correct Recall Rate", fontsize=16, fontweight="bold", labelpad=10)

# 3. Increase tick numbers size and make tick marks thicker
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.tick_params(axis='both', which='major', width=2.5, length=7)

# 4. Make the remaining axes lines (bottom and left) thicker
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)

# 5. Remove top and right lines (spines)
sns.despine(top=True, right=True)

# 6. Ensure grid is completely off
ax.grid(False)

# Clean up the legend (remove the box around it for a cleaner look)
# Code for an INSIDE legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='best', fontsize=14, frameon=True)
plt.tight_layout()

# Save the publication-ready figure
plt.savefig("results/ers_vs_behavior_publication.png", dpi=300, bbox_inches='tight')
plt.show()