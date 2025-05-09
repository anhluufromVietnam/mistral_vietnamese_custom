import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
df = pd.read_csv("llama_benchmark_results.csv")

# Filter out failed runs
df_success = df[df["error"].isna() | (df["error"] == "")]

# Convert to numeric just in case
df_success["ctx"] = pd.to_numeric(df_success["ctx"])
df_success["gpu_layers"] = pd.to_numeric(df_success["gpu_layers"])
df_success["batch"] = pd.to_numeric(df_success["batch"])
df_success["load_time"] = pd.to_numeric(df_success["load_time"])
df_success["response_time"] = pd.to_numeric(df_success["response_time"])

# Set plot style
sns.set(style="whitegrid")

# **Improved Graph: Response Time vs Context Size, Batch Size, and GPU Layers**
plt.figure(figsize=(12, 8))

# We will use a palette for batch size to add more color
palette = sns.color_palette("Set2", n_colors=len(df_success['batch'].unique()))

sns.scatterplot(data=df_success, x="ctx", y="response_time", hue="batch", style="gpu_layers", palette=palette, s=100, markers=["o", "s", "D", "^"], legend="full")

# Titles and labels
plt.title("🕒 Response Time vs Context Size (with Batch Size and GPU Layers)")
plt.xlabel("Context Size (ctx)")
plt.ylabel("Response Time (s)")
plt.legend(title="Batch Size / GPU Layers", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Save and show the plot
plt.savefig("response_time_vs_ctx_batch_gpu_layers.png")
plt.show()
