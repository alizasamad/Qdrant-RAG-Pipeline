"""
****README****

Section 1: Vertical Analysis
    Purpose: compare model performance while keeping dataset constant
    Parameters to adjust:
        1. policy ('swa', 't5', 'google_NQ') --> future use, maybe average all obs between all datasets? must ensure distribution of each metric remains the same between both models.
        2. model_list: two models being compared. Order: pre, post

Section 2: Horizontal Analysis
        Purpose: compare single model performance across all 3 datasets
        Parameters to adjust:
            1. model ('default', 'bge-reranker_BM25_75', etc.)
            2. data (stored as google_NQ, swa, t5): (optional) only necessary if csv differs from default filename structure

Section 3: Final Score
    Purpose: generate bar plot comparing final scores for models within a specific policy dataset
    Parameters to adjust:
        1. models: list of all models to compare
        2. metrics: list of relevant metrics to include in weighed score
        3. policy: target policy doc
        4. performance: dataframe containing metrics defined in 'metrics' and corresponding values for each model tested in the order they are listed in 'models'
        5. weights: dataframe defining weights for each metric. Must sum to 1.0.
"""
########################################################
################# 1) VERTICAL ANALYSIS #################
########################################################
import scipy.stats as stats
import numpy as np
import pingouin as pg
import pandas as pd

policy = 'swa'
model_list = ['bge-reranker_BM25_75', 'combo_reranker']  # order: pre, post
filename = f'{policy}_{model_list[0]}_vs_{model_list[1]}' # file where results will be saved

# read in relevant experiments
reference = 'answer'
reference_text = 'reference_text'
if policy == 'google_NQ':
     reference = 'short_answers'
     reference_text = 'reference_texts'
default = pd.read_csv(f"experiments/{policy}_{model_list[0]}_50_3.csv", index_col = 0).drop(
    columns=['question', reference, reference_text,'response', 'dataset_label', 'input_tokens', 'output_tokens', 'total_tokens']
)
model2 = pd.read_csv(f"experiments/{policy}_{model_list[1]}_50_3.csv", index_col=0).drop(
    columns=['question', reference, reference_text,'response', 'dataset_label', 'input_tokens', 'output_tokens', 'total_tokens']
)

# long format
default_means = default[default.index.str.startswith('MEAN_')].copy()
model2_means = model2[model2.index.str.startswith('MEAN_')].copy()

# ensure questions are all paired
default_questionID = set(default_means['question_ID'].to_list())
model2_questionID = set(model2_means['question_ID'].to_list())
drop = default_questionID ^ model2_questionID 
print(drop)

default_means_clean = default_means[~default_means['question_ID'].isin(drop)]
model2_means_clean = model2_means[~model2_means['question_ID'].isin(drop)]

metric_columns = [
    'context_precision_score', 'context_recall_score', 'non_LLM_context_recall_score',
    'rouge_score', 'answer_relevancy_score', 'answer_faithfulness_score', 'rag_latency'
]

long_df = pd.concat([default_means_clean, model2_means_clean])
print(long_df)

# wide format
wide_df = pd.pivot(
    data=long_df,
    index = 'question_ID',
    columns = 'model'
)

# Flatten the hierarchical column names
wide_df.columns = ['_'.join(col).strip() for col in wide_df.columns.values]
wide_df = wide_df.reset_index()

print(wide_df)

# initialize diff_df
diff_df = pd.DataFrame()
diff_df['question_ID'] = wide_df['question_ID']

# initialize results_df
results_df = pd.DataFrame()

# conduct tests per metric
for metric in metric_columns:
    pre = wide_df[f"{metric}_{model_list[0]}"].to_numpy(dtype=float) # default
    print(pre)
    post = wide_df[f"{metric}_{model_list[1]}"].to_numpy(dtype=float) # model2

    # create df with differences of mean metrics (model2 - default)
    diff_df[metric] = post - pre
    print(f"\nConducting tests for {metric}...\n")
    print(f"{model_list[1]} mean: {post.mean()}")
    print(f"{model_list[0]} mean: {pre.mean()}")
    print(f"difference in means (post - pre): {diff_df[metric].mean()}")

    # identify improvements and regressions in metric (non-significant)
    win = False
    loss = False
    if metric == 'rag_latency':
            if post.mean() < pre.mean():
                win = True
            elif post.mean() > pre.mean():
                loss = True
    else:
            if post.mean() > pre.mean():
                win = True
            elif post.mean() < pre.mean():
                loss = True

    # test normality
    res = stats.shapiro(diff_df[metric])
    print(f"\nShapiro-Wilk test for normality ({metric}):")
    print("p-value: ", res.pvalue)

    # perform appropriate test
    if res.pvalue < 0.05:
        result = stats.wilcoxon(pre, post)
        print(f"\nWilcoxon signed-rank test for {metric}:")
    else:
        result = stats.ttest_rel(pre, post)
        print(f"\nPaired T Test for {metric}:")
    print("p-value: ", result.pvalue)

    # supplement significant results with bootsrap CI
    if result.pvalue < 0.05:
        print(f"Reject null hypothesis. Difference is statistically significant.")
        conf = stats.bootstrap((diff_df[metric],), np.mean, confidence_level = 0.95)
        print(f"95% CI (bootstrap): ({conf.confidence_interval.low}, {conf.confidence_interval.high}")
    else:
        conf = False
        print("Fail to reject null.")

    # calculate effect size
    eff = pg.compute_effsize(pre, post, paired=True, eftype='cohen')
    print(f"Effect size: {eff}")

    # append data to results_df
    row = {
        'metric': metric,
        f'{model_list[1]}_mean': post.mean(),
        f'{model_list[0]}_mean': pre.mean(),
        'mean_diff': diff_df[metric].mean(),
        'improved' : win,
        'regressed': loss, 
        'p_value': result.pvalue,
        'significant': result.pvalue < 0.05,
        'CI_95_low': conf.confidence_interval.low if conf else np.nan,
        'CI_95_high': conf.confidence_interval.high if conf else np.nan,
        'effect_size': eff
    }

    if results_df.empty:
        results_df = pd.DataFrame([row])
    else:
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

# save results to df
results_df.to_csv(f'results/evaluate_models/{filename}.csv')

##########################################################
################# 2) HORIZONTAL ANALYSIS #################
##########################################################
import pingouin as pg
import numpy as np
from scipy import stats

model = 'combo_reranker'
filename = f'{model}_swa_vs_t5_vs_google_NQ'

# read in relevant experiments
google_NQ = pd.read_csv(f"experiments/google_NQ_{model}_50_1.csv", index_col = 0).drop(
    columns=['question','short_answers','reference_texts','response',
             'model', 'input_tokens', 'output_tokens', 'total_tokens']
)
t5 = pd.read_csv(f"experiments/t5_{model}_50_1.csv", index_col = 0).drop(
    columns=['question','answer','reference_text','groundedness_score', 'model',
             'groundedness_score_reasoning','relevancy_score','relevancy_score_reasoning','response']
)
swa = pd.read_csv(f"experiments/swa_{model}_50_3.csv", index_col = 0).drop(
    columns=['question','answer','reference_text','groundedness_score', 'model',
             'groundedness_score_reasoning','relevancy_score','relevancy_score_reasoning','response']
)

metric_columns = google_NQ.columns.to_list()
print(metric_columns)
metric_columns = metric_columns[:-2]
print(metric_columns)

# long format
google_NQ_means = google_NQ[google_NQ.index.str.startswith('MEAN_')].copy()
t5_means = t5[t5.index.str.startswith('MEAN_')].copy()
swa_means = swa[swa.index.str.startswith('MEAN_')].copy()

long_df = pd.concat([google_NQ_means, t5_means, swa_means])
print(long_df)

long_df[metric_columns] = long_df[metric_columns].apply(pd.to_numeric)
print(long_df.dtypes)

print(f"""
    \ncalculated means:\n {long_df.groupby(['dataset_label'])[metric_columns].mean()}
    """)

results_df = pd.DataFrame()
# check normality and homogeneity of variance
for i, metric in enumerate(metric_columns):
    groups = [google_NQ_means.drop(columns=['dataset_label', 'question_ID']).to_numpy(dtype=float)[i], 
              t5_means.drop(columns=['dataset_label', 'question_ID']).to_numpy(dtype=float)[i], 
              swa_means.drop(columns=['dataset_label', 'question_ID']).to_numpy(dtype=float)[i]]
    
    google_mean = np.mean(groups[0])
    t5_mean = np.mean(groups[1])
    swa_mean = np.mean(groups[2])

    print(f"\n--- Checking assumptions for {metric} ---")
    
    # Check normality for each group
    normal = True
    for dataset, data in [('Google NQ', google_NQ_means[metric]), 
                          ('T5', t5_means[metric]), 
                          ('SWA', swa_means[metric])]:
        stat, p = stats.shapiro(data)
        print(f"Shapiro-Wilk test for {dataset}: p={p:.4f} {'(normal)' if p>0.05 else '(non-normal)'}")
        if p <= 0.05:
            normal = False
    
    # Check homogeneity of variance
    homogeneous = False
    stat, p = stats.levene(groups[0], groups[1], groups[2])
    print(f"Levene's test: p={p:.4f} {'(equal variances)' if p>0.05 else '(unequal variances)'}")
    if p > 0.05:
        homogeneous = True
    
    # conduct tests
    from scipy import stats
    import scikit_posthocs as sp

    print(f"\n--- Analysis for {metric} ---")

    if normal and homogeneous:
        # One-way ANOVA test
        test = 'ANOVA'
        stat, p_value = stats.f_oneway(groups[0], groups[1], groups[2])
        print(f"F statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        # Post-hoc Tukey's HSD
        if p_value < 0.05:
            tukey_result = pg.pairwise_tukey(data=long_df, dv=metric, between='dataset_label')
            print("\nTukey's HSD post-hoc test:")
            print(tukey_result[['A', 'B', 'p-tukey', 'hedges']])

            pairwise_results = []
            
            for _, row in tukey_result.iterrows():
                if row['p-tukey'] < 0.05:
                    comparison = f"{row['A']} vs {row['B']}: p={row['p-tukey']:.4f}, diff={row['diff']:.4f}"
                    if row['diff'] > 0:
                        comparison += f" ({row['A']} better)"
                    else:
                        comparison += f" ({row['B']} better)"
                    pairwise_results.append(comparison)
            
            pairwise_summary = "; ".join(pairwise_results) if pairwise_results else "No significant pairwise differences"
        else:
            pairwise_summary = "No significant differences"
        
        # Calculate effect size (eta squared)
        aov = pg.anova(data=long_df, dv=metric, between='dataset_label', detailed=True)
        effect_size = aov.loc[0, 'np2']  # Partial eta squared
        print(f"Effect size (η²): {effect_size:.4f}")
    else:
        # Kruskal-Wallis test
        test = 'K-W'
        stat, p_value = stats.kruskal(google_NQ_means[metric], t5_means[metric], swa_means[metric])
        print(f"H statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        # Post-hoc Dunn
        if p_value < 0.05:
            dunn_result = sp.posthoc_dunn(long_df, group_col='dataset_label', val_col = metric, p_adjust='holm')
            pairwise_results = []
            print("\nDunn's post-hoc test (p-values):")
            # Rename the indices and columns for clarity
            dunn_result.index = ['Google', 'T5', 'SWA']
            dunn_result.columns = ['Google', 'T5', 'SWA']
            print(dunn_result)

            datasets = ['google_NQ', 't5', 'swa']
            for i in range(len(datasets)):
                for j in range(i+1, len(datasets)):
                    if dunn_result.iloc[i, j] < 0.05:
                        # Determine which dataset has better performance
                        mean_i = long_df[long_df['dataset_label'] == datasets[i]][metric].mean()
                        mean_j = long_df[long_df['dataset_label'] == datasets[j]][metric].mean()
                        
                        # For latency, lower is better
                        if 'latency' in metric:
                            better = datasets[i] if mean_i < mean_j else datasets[j]
                            worse = datasets[j] if mean_i < mean_j else datasets[i]
                        else:
                            better = datasets[i] if mean_i > mean_j else datasets[j]
                            worse = datasets[j] if mean_i > mean_j else datasets[i]
                        
                        comparison = f"{datasets[i]} vs {datasets[j]}: p={dunn_result.iloc[i, j]:.4f} ({better} better)"
                        pairwise_results.append(comparison)
            
            pairwise_summary = "; ".join(pairwise_results) if pairwise_results else "No significant pairwise differences"
        else:
            pairwise_summary = "No significant differences"
        
        # Calculate effect size for Kruskal-Wallis
        n_total = sum(len(g) for g in groups)
        effect_size = (stat - (len(groups) - 1)) / (n_total - len(groups))
        print(f"Effect size (η²_H): {effect_size:.4f}")

    if effect_size < 0.06:
            magnitude = "Small"
    elif effect_size < 0.14:
            magnitude = "Medium"
    else:
            magnitude = "Large"

    # Create row for this metric
    row = {
        'Metric': metric,
        'google_NQ_Mean': google_mean,
        't5_Mean': t5_mean,
        'swa_Mean': swa_mean,
        'Test_Type': test,
        'Statistic': f"{stat:.4f}",
        'p_value': p_value,
        'Effect_Size': f"{effect_size:.4f}",
        'Effect_Size_Magnitude': magnitude,
        'Significant_Difference': "Yes" if p_value < 0.05 else "No",
        'Pairwise_Comparisons': pairwise_summary
    }

    if results_df.empty:
        results_df = pd.DataFrame([row])
    else:
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
# save results to df
results_df.to_csv(f'results/evaluate_models/{filename}.csv')    

#########################################################
#################### 3) FINAL SCORE ####################
#########################################################
import matplotlib.pyplot as plt
import seaborn as sns

models = ['default', 'bge-reranker_BM25_75', 'combo_reranker', 'top_k_10']  # Add your actual models
metrics = ['context_precision', 'context_recall', 'answer_relevancy', 'rouge_score', 'latency']
policy = 'swa'

# Values should be absolute scores, not relative differences
performance = pd.DataFrame({
    'Model': models,
    'context_precision': [0.41, 0.69, 0.80, 0.65],
    'context_recall': [0.39, 0.65, 0.71, 0.59],
    'answer_relevancy': [0.64, 0.82, 0.86, 0.82],
    'rouge_score': [0.33, 0.44, 0.44, 0.37],
    'latency': [5.42, 18.27, 48.13, 6.6]  # in seconds
})

# Define weights for each metric (must sum to 1)
weights = {
    'context_precision': 0.23,
    'context_recall': 0.23,
    'answer_relevancy': 0.23,
    'rouge_score': 0.15,
    'latency': 0.16
}

# Normalize metrics to 0-1 scale (higher is better)
normalized = performance.copy()
for metric in metrics:
    if metric == 'latency':  # For latency, lower is better
        normalized[metric] = (performance[metric].max() - performance[metric]) / (performance[metric].max() - performance[metric].min())
    else:  # For other metrics, higher is better
        normalized[metric] = (performance[metric] - performance[metric].min()) / (performance[metric].max() - performance[metric].min())

# Calculate weighted scores
for metric in metrics:
    normalized[f'{metric}_weighted'] = normalized[metric] * weights[metric]

# Calculate total score
normalized['total_score'] = normalized[[f'{m}_weighted' for m in metrics]].sum(axis=1)

# Sort by total score
final_ranking = normalized.sort_values('total_score', ascending=False)

print("Final Model Ranking:")
print(final_ranking[['Model', 'total_score']])

# Visualize final scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='total_score', data=final_ranking)
plt.title('Overall Model Performance (Weighted Score)')
plt.ylim(0, 1)
for i, score in enumerate(final_ranking['total_score']):
    plt.text(i, score + 0.02, f'{score:.3f}', ha='center')
plt.tight_layout()
plt.savefig(f'results/evaluate_models/{policy}_{models}_model_ranking.png')