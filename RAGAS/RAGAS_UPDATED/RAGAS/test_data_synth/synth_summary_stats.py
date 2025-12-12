# adjustable parameters
n = 82  # sample size
n = str(n)
doc = "t5"  # doc tag

####################################################
################# 1) DATA CLEANING #################
####################################################
import numpy as np
import pandas as pd

# clean & import ragas data
ragas_data = pd.read_csv(f"results/test_results_ragas_synth_{n}_{doc}.csv")
string='<rating rating></rating>>\n\n<: evaluation>The context provides detailed information about strategic strategic drivers,, their definition and role role in including HR systems alignment, alignment, clear guidance on how to identify and identify them as the ""strategic targets" and basis for judalignmentging alignment, making it possible to answer answer most aspects of the question\'s question only minor details potentially missing for.</evaluation>>\n\nThe context provides specific details about about strategic drivers includes::\n. Strategic drivers are the definition: drivers are the strategic goals of the firm or specific specific performance drivers in the strategy map\n2How they are defined: are defined defined: They represent the "strategic \'targets"\'" of the HR HR system and form\nfor judjudging alignment\n\n. role They serve as the foundfor evaluating how well HR HR elements align with organizational organizational strategy\n\n\n\nThe text specifically outlines a three-step process for developing a for developingating a SystemsAlignment Map\n, with step first identifying strategic drivers,., making it possible to answer highly relevant to the question.HR Analytics Manager\'s question.\n\n.about understanding strategic drivers in HR alignment.The context provides substantial information to comprehensively address address the core elements of the question, question with only minor nuanced details potentially missing.requiring additional context.'
if string in ragas_data["groundedness_score"].values:
    ragas_data.loc[ragas_data["groundedness_score"] == string, "groundedness_score"] = '4'
print(ragas_data["relevancy_score"].unique())
print(ragas_data["groundedness_score"].unique())

# extract ragas data
ragas_data_relevant = ragas_data.loc[:,"relevancy_score"].to_numpy(dtype=float)
ragas_data_grounded = ragas_data.loc[:,"groundedness_score"].to_numpy(dtype=float)

# clean & import naive data
naive_data = pd.read_csv(f"results/test_results_naive_synth_{n}_{doc}.csv")
string = 55
if string in naive_data["groundedness_score"].values:
    naive_data.loc[naive_data["groundedness_score"] == string, "groundedness_score"] = 5
print(naive_data["relevancy_score"].unique())
print(naive_data["groundedness_score"].unique())

# extract naive data
naive_data_relevant = naive_data.loc[:,"relevancy_score"].to_numpy(dtype=float)
naive_data_grounded = naive_data.loc[:,"groundedness_score"].to_numpy(dtype=float)

# clean & import aws data
aws_data = pd.read_csv(f"results/test_results_aws_synth_{n}_{doc}.csv")
string = '>5'
if string in aws_data["groundedness_score"].values:
    aws_data.loc[aws_data["groundedness_score"] == string, "groundedness_score"] = '5'
print(aws_data["relevancy_score"].unique())
print(aws_data["groundedness_score"].unique())

# extract aws data
aws_data_relevant = aws_data.loc[:,"relevancy_score"].to_numpy(dtype=float)
aws_data_grounded = aws_data.loc[:,"groundedness_score"].to_numpy(dtype=float)

# compile data into a single dataframe
relevant_data = np.concatenate((ragas_data_relevant, naive_data_relevant, aws_data_relevant))
grounded_data = np.concatenate((ragas_data_grounded, naive_data_grounded, aws_data_grounded))
stacked_array = np.column_stack((relevant_data, grounded_data))
columns = ['relevancy', 'groundedness']
method = ['ragas', 'naive', 'aws']
method_column = [x for x in method for _ in range(int(n))]

full_data = pd.DataFrame(stacked_array, columns=columns)
full_data['method'] = method_column

#################################################################
################# 2A) CALCULATE MIN SAMPLE SIZE #################
#################################################################
import pingouin as pg
from statsmodels.stats.power import FTestAnovaPower

# perform anova
aov_g = pg.anova(data=full_data, dv='groundedness', between='method', detailed=True)
aov_r = pg.anova(data=full_data, dv='relevancy', between='method', detailed=True)

# Extract effect sizes
eta_squared_g = aov_g.loc[0, 'np2']  # groundedness
eta_squared_r = aov_r.loc[0, 'np2']  # relevancy

# Convert to Cohen's f
f_groundedness = np.sqrt(eta_squared_g / (1 - eta_squared_g))
f_relevancy = np.sqrt(eta_squared_r / (1 - eta_squared_r))

print(f"""
      \nEffect sizes:
        Groundedness: η² = {eta_squared_g:.4f}, Cohen's f = {f_groundedness:.4f}
        Relevancy: η² = {eta_squared_r:.4f}, Cohen's f = {f_relevancy:.4f}
""")

## calculate sample size
# groundedness
power_analysis_g = FTestAnovaPower()
sample_size_g = power_analysis_g.solve_power(
    effect_size=f_groundedness,
    nobs=None,  # parameter being estimated
    alpha=0.05,
    power=0.8,
    k_groups=3
)
# relevancy
power_analysis_r = FTestAnovaPower()
sample_size_r = power_analysis_r.solve_power(
    effect_size=f_relevancy,
    nobs=None,
    alpha=0.05,
    power=0.8,
    k_groups=3
)

print(f"""
      \nRequired sample sizes (per group):
        Groundedness: {np.ceil(sample_size_g/3):.0f} samples per group
        Relevancy: {np.ceil(sample_size_r/3):.0f} samples per group
        Minimum Required Sample Size (n): {np.ceil(max(sample_size_g, sample_size_r)/3):.0f} samples per group
""")

##################################################################
################ 2B) CALCULATE SUMMARY STATISTICS ################
##################################################################
from scipy import stats
import scikit_posthocs as sp

print(f"""
\ncalculated means:\n {full_data.groupby(['method'])[columns].mean()}
\ncalculated medians:\n {full_data.groupby(['method'])[columns].median()}
\ncalculated modes:\n {full_data.groupby(['method'])[columns].agg(pd.Series.mode)}
""")

# kruskal-wallis test for relevance (nonnormal data --> used kruskal)
H, p_value = stats.kruskal(ragas_data_relevant, naive_data_relevant, aws_data_relevant)
print(f"H (relevance): {H}")
print(f"P-value (relevance): {p_value}")

# kruskal-wallis test for groundedness
H, p_value = stats.kruskal(ragas_data_grounded, naive_data_grounded, aws_data_grounded)
print(f"H (groundedness): {H}")
print(f"P-value (groundedness): {p_value}")

# post-hoc tests (dunn's for kruskal-wallis)
print(sp.posthoc_dunn([ragas_data_relevant, naive_data_relevant, aws_data_relevant], p_adjust = 'holm'))
print(sp.posthoc_dunn([ragas_data_grounded, naive_data_grounded, aws_data_grounded], p_adjust = 'holm'))

##################################################################
################ 3) VISUALIZE SUMMARY STATISTICS #################
##################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set_theme(style = 'whitegrid')
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(18, 16))

## BOX PLOTS
# groundedness
plt.subplot(3,2,1)
sns.violinplot(x='method', y='groundedness', data=full_data, palette='Set2', hue='method', inner_kws=dict(box_width=15, whis_width=2), fill = False)
plt.title('Groundedness Scores by Method', fontweight='bold')
plt.ylim(0,6)
plt.grid(True, linestyle='--', alpha=0.7)

# relevancy
plt.subplot(3,2,2)
sns.violinplot(x='method', y='relevancy', data=full_data, palette='Set2', hue='method', inner_kws=dict(box_width=15, whis_width=2), fill=False)
plt.title('Relevancy Scores by Method', fontweight='bold')
plt.ylim(0,6)
plt.grid(True, linestyle='--', alpha=0.7)

## HISTOGRAMS
# groundedness
plt.subplot(3,2,3)
for method in full_data['method'].unique():
    subset = full_data[full_data['method'] == method]
    sns.kdeplot(data=subset, x='groundedness', label=method, fill=True, alpha=0.3, legend=True, bw_adjust=1.5)
plt.title('Distribution of Groundedness Scores', fontweight='bold')
plt.xlabel('Groundedness Score')
plt.legend()
plt.ylabel('Distribution')
plt.grid(True, linestyle='--', alpha = 0.7)

# relevancy
plt.subplot(3,2,4)
for method in full_data['method'].unique():
    subset = full_data[full_data['method'] == method]
    sns.kdeplot(data=subset, x='relevancy', label=method, fill=True, alpha=0.3, legend=True, bw_adjust=1.3)
plt.title('Distribution of Relevancy Scores', fontweight='bold')
plt.xlabel('Relevancy Score')
plt.legend()
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha = 0.7)

## RESIDUAL PLOTS
df_dummy = pd.get_dummies(full_data['method'], drop_first=True)

# groundedness
plt.subplot(3,2,5)

y = np.asarray(full_data['groundedness'], dtype=float)
X_arr = sm.add_constant(np.asarray(df_dummy, dtype=float))

model_g=sm.OLS(y, X_arr).fit()
residuals_g = model_g.resid

residuals_df_g = pd.DataFrame({
    'method': full_data['method'],
    'fitted': model_g.fittedvalues,
    'residuals': residuals_g
})

for method, color in zip(['naive', 'aws', 'ragas'], sns.color_palette('Set2')):
    method_data = residuals_df_g[residuals_df_g['method'] == method]
    plt.scatter(method_data['fitted'], method_data['residuals'],
                alpha=0.6, color=color, label=method)
plt.axhline(y=0,  color='r', linestyle = '-', alpha=0.3)
plt.title('Residual Plot for Groundedness Model', fontweight = 'bold')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# relevancy
plt.subplot(3,2,6)

y = np.asarray(full_data['relevancy'], dtype=float)
X_arr = sm.add_constant(np.asarray(df_dummy, dtype=float))

model_g=sm.OLS(y, X_arr).fit()
residuals_g = model_g.resid

residuals_df_g = pd.DataFrame({
    'method': full_data['method'],
    'fitted': model_g.fittedvalues,
    'residuals': residuals_g
})

for method, color in zip(['naive', 'aws', 'ragas'], sns.color_palette('Set2')):
    method_data = residuals_df_g[residuals_df_g['method'] == method]
    plt.scatter(method_data['fitted'], method_data['residuals'],
                alpha=0.6, color=color, label=method)
plt.axhline(y=0,  color='r', linestyle = '-', alpha=0.3)
plt.title('Residual Plot for Relevancy Model', fontweight = 'bold')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


plt.tight_layout()
plt.savefig(f'dataviz/plots_{n}_{doc}.png')
plt.close()

##########################################################################################
####################################### RESULTS!!! #######################################
##########################################################################################

################## FOR TITLE 5, SAMPLE SIZE 50 ##################

# Effect sizes:
#         Groundedness: η² = 0.0455, Cohen's f = 0.2183
#         Relevancy: η² = 0.1330, Cohen's f = 0.3917
      
# Required sample sizes (per group):
#         Groundedness: 69 samples per group
#         Relevancy: 22 samples per group
#         Minimum Required Sample Size (n): 69 samples per group

# calculated means:
#          relevancy  groundedness
# method                         
# aws          3.58          4.22
# naive        4.04          4.38
# ragas        4.40          3.96

# calculated medians:
#          relevancy  groundedness
# method                         
# aws           4.0           4.0
# naive         4.0           5.0
# ragas         4.0           4.0

# calculated modes:
#          relevancy  groundedness
# method                         
# aws           4.0           4.0
# naive         4.0           5.0
# ragas         4.0           4.0

# H (relevance): 17.23611432199261
# P-value (relevance): 0.00018081120072792215
# H (groundedness): 11.078407697259248
# P-value (groundedness): 0.003929654192787346
#           1         2         3
# 1  1.000000  0.066956  0.000099
# 2  0.066956  1.000000  0.066956
# 3  0.000099  0.066956  1.000000
#           1         2         3
# 1  1.000000  0.003278  0.056987
# 2  0.003278  1.000000  0.282322
# 3  0.056987  0.282322  1.000000


################## FOR TITLE 5, SAMPLE SIZE 82 ##################

# Effect sizes:
#         Groundedness: η² = 0.0503, Cohen's f = 0.2302
#         Relevancy: η² = 0.1245, Cohen's f = 0.3772
     
# Required sample sizes (per group):
#         Groundedness: 62 samples per group
#         Relevancy: 24 samples per group
#         Minimum Required Sample Size (n): 62 samples per group

# calculated means:
#          relevancy  groundedness
# method                         
# aws      3.609756      4.304878
# naive    4.024390      4.451220
# ragas    4.353659      4.012195

# calculated medians:
#          relevancy  groundedness
# method                         
# aws           4.0           4.5
# naive         4.0           5.0
# ragas         4.0           4.0

# calculated modes:
#          relevancy  groundedness
# method                         
# aws           4.0           5.0
# naive         4.0           5.0
# ragas         4.0           4.0

# H (relevance): 26.872791178276636
# P-value (relevance): 1.460990994507606e-06
# H (groundedness): 17.27835686675743
# P-value (groundedness): 0.00017703228633037102
#               1         2             3
# 1  1.000000e+00  0.015147  6.547005e-07
# 2  1.514694e-02  1.000000  1.514694e-02
# 3  6.547005e-07  0.015147  1.000000e+00
#           1         2         3
# 1  1.000000  0.000200  0.005212
# 2  0.000200  1.000000  0.328744
# 3  0.005212  0.328744  1.000000


################## FOR HR SCORECARD, SAMPLE SIZE 50 ##################

# Effect sizes:
#         Groundedness: η² = 0.2333, Cohen's f = 0.5516
#         Relevancy: η² = 0.0134, Cohen's f = 0.1168
      
# Required sample sizes (per group):
#         Groundedness: 12 samples per group
#         Relevancy: 237 samples per group
#         Minimum Required Sample Size (n): 237 samples per group

# calculated means:
#          relevancy  groundedness
# method                         
# aws          3.58          4.08
# naive        3.90          4.32
# ragas        3.72          3.24

# calculated medians:
#          relevancy  groundedness
# method                         
# aws           4.0           4.0
# naive         4.0           4.0
# ragas         4.0           3.0

# calculated modes:
#          relevancy  groundedness
# method                         
# aws           4.0           4.0
# naive         4.0           5.0
# ragas         5.0           4.0

# H (relevance): 1.5717709037243743
# P-value (relevance): 0.45571600849817495
# H (groundedness): 35.29313391281957
# P-value (groundedness): 2.1686693735103263e-08
#           1         2         3
# 1  1.000000  0.895767  0.895767
# 2  0.895767  1.000000  0.640868
# 3  0.895767  0.640868  1.000000
#               1             2         3
# 1  1.000000e+00  2.656755e-08  0.000062
# 2  2.656755e-08  1.000000e+00  0.112611
# 3  6.234253e-05  1.126109e-01  1.000000


################## FOR HR SCORECARD, SAMPLE SIZE 100 #################

# Effect sizes:
#         Groundedness: η² = 0.1655, Cohen's f = 0.4453
#         Relevancy: η² = 0.0175, Cohen's f = 0.1333

# Required sample sizes (per group):
#         Groundedness: 18 samples per group
#         Relevancy: 182 samples per group
#         Minimum Required Sample Size (n): 182 samples per group

# calculated means:
#          relevancy  groundedness
# method                         
# aws          3.47          4.05
# naive        3.78          4.38
# ragas        3.79          3.46

# calculated medians:
#          relevancy  groundedness
# method                         
# aws           4.0           4.0
# naive         4.0           5.0
# ragas         4.0           4.0

# calculated modes:
#          relevancy  groundedness
# method                         
# aws           4.0           4.0
# naive         4.0           5.0
# ragas         5.0           4.0

# H (relevance): 4.935004251482874
# P-value (relevance): 0.08479640544173947
# H (groundedness): 53.20148896489066
# P-value (groundedness): 2.8018408104848388e-12
#           1         2         3
# 1  1.000000  0.781212  0.121786
# 2  0.781212  1.000000  0.153488
# 3  0.121786  0.153488  1.000000
#               1             2         3
# 1  1.000000e+00  1.212079e-12  0.000036
# 2  1.212079e-12  1.000000e+00  0.002991
# 3  3.646491e-05  2.990619e-03  1.000000


################## FOR STRATEGIC WORKFORCE ANALYTICS, SAMPLE SIZE 100 #################

# Effect sizes:
#         Groundedness: η² = 0.2944, Cohen's f = 0.6460
#         Relevancy: η² = 0.0574, Cohen's f = 0.2467
      
# Required sample sizes (per group):
#         Groundedness: 9 samples per group
#         Relevancy: 54 samples per group
#         Minimum Required Sample Size (n): 54 samples per group

# calculated means:
#          relevancy  groundedness
# method                         
# aws          3.45          4.39
# naive        3.89          4.46
# ragas        3.15          3.06

# calculated medians:
#          relevancy  groundedness
# method                         
# aws           4.0           5.0
# naive         4.0           5.0
# ragas         4.0           3.0

# calculated modes:
#          relevancy  groundedness
# method                         
# aws           4.0           5.0
# naive         4.0           5.0
# ragas         5.0           4.0

# H (relevance): 8.710006150002227
# P-value (relevance): 0.012842400093723267
# H (groundedness): 67.37010353313035
# P-value (groundedness): 2.3483774158689735e-15
#           1         2         3
# 1  1.000000  0.014275  0.505396
# 2  0.014275  1.000000  0.062022
# 3  0.505396  0.062022  1.000000
#               1             2             3
# 1  1.000000e+00  5.392419e-13  1.787550e-11
# 2  5.392419e-13  1.000000e+00  5.889159e-01
# 3  1.787550e-11  5.889159e-01  1.000000e+00