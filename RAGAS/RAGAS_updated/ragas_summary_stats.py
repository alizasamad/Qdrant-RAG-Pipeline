"""
visualize initial benchmark: bar chart w/ SME + histogram?; stack bar chart comparing magnitude of std (LLM consistency & sample variatin), percent contribution in parentheses
perform paired t test to compare benchmark model to all updates
use ANOVA  + Post-hoc tests for all models that outperform base (tukeyHSD)
visualize statistically significant differences (radar/spider chart)
"""
# adjustable parameters
filename = 'swa_default_50_3'
end_index = 49
compare = ['t5_default_50_3','google_NQ_default_50_3']
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
# import matplotlib.pyplot as plt
# import seaborn as sns

# # bar plot for mean w error bars
# plt.figure(figsize=(16,10))
# x = np.arange(len(columns))

# plt.bar(
#     x, 
#     mean_metrics,
#     yerr = conf_int,
#     error_kw = {'ecolor': 'black', 'elinewidth': 2, 'capsize': 5},
#     color = 'lightblue'
#     )
# plt.title(f"RAGAS Metrics: {filename}", fontsize=16)
# plt.xlabel('Metrics', fontsize=14)
# plt.ylabel('Score (0-1 Scale)', fontsize=14)
# plt.xticks(x, columns, rotation=45, ha='right', fontsize=10)
# plt.tight_layout()
# plt.savefig(f'results/summary_stats/{filename}_mean_plot.png')
# plt.close()

# # stacked bar chart for variance
# plt.figure(figsize=(14,6))

# var_btwn_abs = summary_df['std_btwn_samples'] ** 2  # Between-observation variance
# var_wtn_abs = summary_df['mean_std_wtn_sample'] ** 2  # Within-observation (LLM) variance
# total_var = var_btwn_abs + var_wtn_abs

# var_btwn_pct = (var_btwn_abs / total_var * 100).values
# var_wtn_pct = (var_wtn_abs / total_var * 100).values

# plt.bar(x, var_btwn_abs, color = 'coral', label = 'Between Samples (%)')
# plt.bar(x, var_wtn_abs, bottom = var_btwn_abs, color = 'lightblue', label = 'Within Sample (LLM) (%)')

# for i, (vb_abs, vw_abs, vb_pct, vw_pct) in enumerate(zip(var_btwn_abs, var_wtn_abs, var_btwn_pct, var_wtn_pct)):
#     plt.text(i, vb_abs/2, f"{vb_pct:.1f}%", ha='center', va='center', color='black', fontweight='bold')
#     plt.text(i, vb_abs + vw_abs/2, f"{vw_pct:.1f}%", ha='center', va='center', color = 'black', fontweight = 'bold')

# plt.title("Variance Components in RAGAS Metrics", fontsize = 16)
# plt.xlabel('Metrics', fontsize = 14)
# plt.ylabel('Absolute Variance', fontsize= 14)
# plt.xticks(x, columns, rotation=45, ha='right', fontsize=10)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'results/summary_stats/{filename}_var_components.png')
# plt.close()

# # visualize distributions
# plt.figure(figsize=(16,10))

# mean_df = pd.DataFrame(mean_data, columns=columns)
# for i, col in enumerate (columns):
#     plt.subplot(3,3,i+1)
#     sns.violinplot(y=mean_df[col], inner_kws=dict(box_width=15, whis_width=2), fill = False, color='coral')
#     plt.axhline(mean_metrics[i], color = 'red', linestyle = '--', label = f"Mean: {mean_metrics[i]:.4f}")
#     plt.axhspan(
#         mean_metrics[i] - conf_int[i],
#         mean_metrics[i] + conf_int[i],
#         alpha = 0.2, color= 'red', label = '95% CI'
#     )

#     plt.title(f"{col} Distribution")
#     plt.xlabel("Score")
#     plt.grid(True, linestyle='--', alpha=0.3)

#     if i == 0:
#         plt.legend(loc='lower right')

# plt.tight_layout()
# plt.savefig(f'results/summary_stats/{filename}_distributions.png')
# plt.close()

# visualize combined plot
import matplotlib.pyplot as plt

if compare:
    print(compare)

    data2 = pd.read_csv(f"results/summary_stats/{compare[0]}_summary_stats.csv", index_col=0)
    mean_metrics2 = data2['mean'].to_numpy(dtype=float)
    conf_int2 = mean_metrics2 - data2['ci_95_lower'].to_numpy(dtype=float)
    data2 = pd.read_csv(f"results/summary_stats/{compare[0]}_summary_stats.csv", index_col=0)
    mean_metrics2 = data2['mean'].to_numpy(dtype=float)
    conf_int2 = mean_metrics2 - data2['ci_95_lower'].to_numpy(dtype=float)

    data3 = pd.read_csv(f"results/summary_stats/{compare[1]}_summary_stats.csv", index_col=0)
    mean_metrics3 = data3['mean'].to_numpy(dtype=float)
    conf_int3 = mean_metrics3 - data3['ci_95_lower'].to_numpy(dtype=float)
    data3 = pd.read_csv(f"results/summary_stats/{compare[1]}_summary_stats.csv", index_col=0)
    mean_metrics3 = data3['mean'].to_numpy(dtype=float)
    conf_int3 = mean_metrics3 - data3['ci_95_lower'].to_numpy(dtype=float)

    # data4 = pd.read_csv(f"results/summary_stats/{compare[2]}_summary_stats.csv", index_col=0)
    # mean_metrics4 = data4['mean'].to_numpy(dtype=float)
    # conf_int4 = mean_metrics4 - data4['ci_95_lower'].to_numpy(dtype=float)
    # data4 = pd.read_csv(f"results/summary_stats/{compare[2]}_summary_stats.csv", index_col=0)
    # mean_metrics4 = data4['mean'].to_numpy(dtype=float)
    # conf_int4 = mean_metrics4 - data4['ci_95_lower'].to_numpy(dtype=float)


    plt.figure(figsize=(16,10))
    w, x = 0.25, np.arange(len(columns))
    plt.bar(
        x - w, 
        mean_metrics,
        w,
        yerr = conf_int,
        error_kw = {'ecolor': 'black', 'elinewidth': 2, 'capsize': 5},
        color = 'lightblue',
        label = filename
        )
    plt.bar(
        x, 
        mean_metrics2,
        w,
        yerr = conf_int2,
        error_kw = {'ecolor': 'black', 'elinewidth': 2, 'capsize': 5},
        color = 'lightgreen',
        label = compare[0]
        )
    plt.bar(
        x + w, 
        mean_metrics3,
        w,
        yerr = conf_int3,
        error_kw = {'ecolor': 'black', 'elinewidth': 2, 'capsize': 5},
        color = 'coral',
        label = compare[1]
        )
    # plt.bar(
    #     x + w * (3/2), 
    #     mean_metrics4,
    #     w,
    #     yerr = conf_int4,
    #     error_kw = {'ecolor': 'black', 'elinewidth': 2, 'capsize': 5},
    #     color = 'lightpink',
    #     label = compare[2]
    #     )
    plt.title(f"RAGAS Metrics: {filename} vs {compare}", fontsize=16)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Score (0-1 Scale)', fontsize=14)
    plt.xticks(x, columns, rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/summary_stats/{filename}_vs_{compare}_mean_plot.png')
    plt.close()
