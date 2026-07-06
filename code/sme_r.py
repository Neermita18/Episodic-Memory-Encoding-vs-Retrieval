import pandas as pd
import statsmodels.formula.api as smf

# Load SME data
df_sme = pd.read_csv(r"C:\Users\Neermita\Desktop\memory_and_task\code\results\sme_metrics.csv").dropna(subset=["acc"])

# Center accuracy around chance level (0.50)
df_sme["acc_above_chance"] = df_sme["acc"] - 0.50

# Run a Mixed-Effects Model testing just the intercept
model_sme = smf.mixedlm("acc_above_chance ~ 1", df_sme, groups=df_sme["subject"])
result_sme = model_sme.fit()
print("--- SME Above-Chance Results ---")
print(result_sme.summary())