"""
Notes:
visualize initial benchmark: bar chart w/ SME + histogram?; stack bar chart comparing magnitude of std (LLM consistency & sample variatin), percent contribution in parentheses
perform paired t test to compare benchmark model to all updates
use ANOVA  + Post-hoc tests for all models that outperform base (tukeyHSD)
visualize statistically significant differences (radar/spider chart)


****READ ME****
MANDATORY: must adjust the following parameters:
    1. filename - target file to clean & calculate summary stats
    2. end_index - index of final obs (should be <=49)
    3. compare - (optional) list of target files or 'False'
        **Note: must clean & create summary stats df (sections 1 & 2)  for all 'compare' datasets before running section 3
    4. compared_indices - (optional) indices of final obs for 'compare' datasets
"""
# adjustable parameters
filename = 'swa_default_50_3'
end_index = 49
compare = ['t5_default_50_3','google_NQ_default_50_3'] # or False
compare_indices = [49, 47]

####################################################
################# 1) DATA CLEANING #################
####################################################
import pandas as pd
import numpy as np

# import data
data = pd.read_csv(f"experiments/{filename}.csv", index_col=0)
data_filtered = data[data['rouge_score'].notna()]

mean_data = data_filtered.loc['MEAN_0': f'MEAN_{str(end_index)}', 'rag_latency':'answer_faithfulness_score']
mean_data = mean_data.to_numpy(dtype = float)

std_data = data_filtered.loc['STD_0':f'STD_{str(end_index)}', 'rag_latency':'answer_faithfulness_score'].to_numpy(dtype=float)
columns = ["rag_latency", "context_precision_score","context_recall_score", 
                    "non_LLM_context_recall_score", "rouge_score", "answer_relevancy_score", 
                    "answer_faithfulness_score"]

# normalize means
original_latency_means = mean_data[:, 0].copy()
latency_m = [arr[0] for arr in mean_data]
min_val = min(latency_m)
max_val = max(latency_m)

for i in range(len(mean_data)):
    norm_latency_m = (mean_data[i][0] - min_val) / (max_val-min_val)
    mean_data[i][0] = norm_latency_m

# normalize latency for std
for i in range(len(std_data)):
    std_data[i][0] = std_data[i][0] / original_latency_means[i]

###################################################################
################# 2) CALCULATE SUMMARY STATISTICS #################
###################################################################
from scipy import stats

# calculate summary statistics
mean_metrics = np.mean(mean_data, axis=0)
print(mean_metrics)
std_of_the_mean = np.std(mean_data, axis=0)  # variance BETWEEN sample means
print(std_of_the_mean)
std_metrics_LLM = np.mean(std_data, axis=0)  # variance WITHIN sample metrics
print(std_metrics_LLM)

# calculate confidence interval
n = len(mean_data)
print(n)
conf_int = stats.t.ppf(0.975, n-1) * std_of_the_mean / np.sqrt(n)
print(conf_int)

# compile stats
summary_df = pd.DataFrame({
    'metric': columns,
    'mean': mean_metrics,
    'std_btwn_samples': std_of_the_mean,
    'mean_std_wtn_sample': std_metrics_LLM,
    'ci_95_lower': mean_metrics - conf_int,
    'ci_95_upper': mean_metrics + conf_int,
})

summary_df['cv_btwn_samples'] = summary_df['std_btwn_samples'] / summary_df['mean']
summary_df['cv_wtn_sample'] = summary_df['mean_std_wtn_sample'] / summary_df['mean']

print("Summary Statistics:")
print(summary_df)

summary_df.to_csv(f'results/summary_stats/{filename}_summary_stats.csv')

# ##################################################################
# ################ 3) VISUALIZE SUMMARY STATISTICS #################
# ##################################################################
import matplotlib.pyplot as plt
import seaborn as sns

# bar plot for mean w error bars
plt.figure(figsize=(16,10))
x = np.arange(len(columns))

plt.bar(
    x, 
    mean_metrics,
    yerr = conf_int,
    error_kw = {'ecolor': 'black', 'elinewidth': 2, 'capsize': 5},
    color = 'lightblue'
    )
plt.title(f"RAGAS Metrics: {filename}", fontsize=16)
plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Score (0-1 Scale)', fontsize=14)
plt.xticks(x, columns, rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(f'results/summary_stats/{filename}_mean_plot.png')
plt.close()

# stacked bar chart for variance
plt.figure(figsize=(14,6))

var_btwn_abs = summary_df['std_btwn_samples'] ** 2  # Between-observation variance
var_wtn_abs = summary_df['mean_std_wtn_sample'] ** 2  # Within-observation (LLM) variance
total_var = var_btwn_abs + var_wtn_abs

var_btwn_pct = (var_btwn_abs / total_var * 100).values
var_wtn_pct = (var_wtn_abs / total_var * 100).values

plt.bar(x, var_btwn_abs, color = 'coral', label = 'Between Samples (%)')
plt.bar(x, var_wtn_abs, bottom = var_btwn_abs, color = 'lightblue', label = 'Within Sample (LLM) (%)')

for i, (vb_abs, vw_abs, vb_pct, vw_pct) in enumerate(zip(var_btwn_abs, var_wtn_abs, var_btwn_pct, var_wtn_pct)):
    plt.text(i, vb_abs/2, f"{vb_pct:.1f}%", ha='center', va='center', color='black', fontweight='bold')
    plt.text(i, vb_abs + vw_abs/2, f"{vw_pct:.1f}%", ha='center', va='center', color = 'black', fontweight = 'bold')

plt.title("Variance Components in RAGAS Metrics", fontsize = 16)
plt.xlabel('Metrics', fontsize = 14)
plt.ylabel('Absolute Variance', fontsize= 14)
plt.xticks(x, columns, rotation=45, ha='right', fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig(f'results/summary_stats/{filename}_var_components.png')
plt.close()

# Visualize combined plot
if compare:
    print("Comparing with:", compare)
    
    plt.figure(figsize=(16, 10))

    # define plot params
    num_datasets = 1 + len(compare)  
    w = 0.8 / num_datasets
    x = np.arange(len(columns))
    colors = ['lightblue', 'lightgreen', 'coral', 'lightpink', 'lightskyblue']
    
    all_data = []
    all_labels = [filename] + compare
    
    # Load target dataset metrics
    target_data = pd.read_csv(f"results/summary_stats/{filename}_summary_stats.csv", index_col=0)
    target_means = target_data['mean'].to_numpy(dtype=float)
    target_conf = target_means - target_data['ci_95_lower'].to_numpy(dtype=float)
    all_data.append((target_means, target_conf))
    
    # Load comparison datasets
    for comp_file in compare:
        comp_data = pd.read_csv(f"results/summary_stats/{comp_file}_summary_stats.csv", index_col=0)
        comp_means = comp_data['mean'].to_numpy(dtype=float)
        comp_conf = comp_means - comp_data['ci_95_lower'].to_numpy(dtype=float)
        all_data.append((comp_means, comp_conf))
    
    # Place bars at correct positions
    for i, ((means, conf_interval), label, color) in enumerate(zip(all_data, all_labels, colors)):
        position = x + w * (i - (num_datasets - 1) / 2)
        
        plt.bar(
            position,
            means,
            width=w,
            yerr=conf_interval,
            error_kw={'ecolor': 'black', 'elinewidth': 2, 'capsize': 5},
            color=color,
            label=label
        )
    
    # Set titles and labels
    if len(compare) > 1:
        comp_string = ", ".join(compare)
    else:
        comp_string = compare[0]
        
    plt.title(f"RAGAS Metrics: {filename} vs {comp_string}", fontsize=16)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Score (0-1 Scale)', fontsize=14)
    plt.xticks(x, columns, rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/summary_stats/{filename}_vs_{"_".join(compare)}_mean_plot.png')
    plt.close()