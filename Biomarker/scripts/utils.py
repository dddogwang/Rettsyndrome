import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu


# 删除离群点
def remove_outliers(df, multiplier=1.5):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - multiplier * IQR)) | (df > (Q3 + multiplier * IQR))).any(axis=1)]

def loadcsv(loadpath_RETT, loadpath_CTRL):

    # 读取 CSV 文件
    df_RETT = pd.read_csv(loadpath_RETT).dropna()  # 删除包含 NaN 的样本
    df_CTRL = pd.read_csv(loadpath_CTRL).dropna()  # 删除包含 NaN 的样本
    print(f"LOAD {loadpath_RETT} {df_RETT.shape}")
    print(f"LOAD {loadpath_CTRL} {df_CTRL.shape}")
    
#     # 删除离群点
#     df_RETT_filtered = remove_outliers(df_RETT_scaled, multiplier=2)
#     df_CTRL_filtered = remove_outliers(df_CTRL_scaled, multiplier=2)
#     print(f"remove outliers {df_RETT_filtered.shape}, {df_CTRL_filtered.shape}")

#   添加状态标签
    df_RETT['State'] = 'RETT'
    df_CTRL['State'] = 'CTRL'
    
    # 合并数据
    df_combined = pd.concat([df_CTRL, df_RETT])
    
    return df_combined, df_RETT, df_CTRL

def loadcsv_Standard(loadpath_RETT, loadpath_CTRL):

    # 读取 CSV 文件
    df_RETT = pd.read_csv(loadpath_RETT).dropna()  # 删除包含 NaN 的样本
    df_CTRL = pd.read_csv(loadpath_CTRL).dropna()  # 删除包含 NaN 的样本
    print(f"LOAD {loadpath_RETT} {df_RETT.shape}")
    print(f"LOAD {loadpath_CTRL} {df_CTRL.shape}")
    
    # 标准化数据
    scaler = StandardScaler()
    df_RETT_scaled = scaler.fit_transform(df_RETT)
    df_CTRL_scaled = scaler.fit_transform(df_CTRL)
    # 将标准化后的数据转换回 DataFrame
    df_RETT_scaled = pd.DataFrame(df_RETT_scaled, columns=df_RETT.columns)
    df_CTRL_scaled = pd.DataFrame(df_CTRL_scaled, columns=df_CTRL.columns)

    # 添加状态标签
    df_RETT = df_RETT_scaled.copy()
    df_CTRL = df_CTRL_scaled.copy()
    df_RETT['State'] = 'RETT'
    df_CTRL['State'] = 'CTRL'
    
    # 合并数据
    df_combined = pd.concat([df_CTRL, df_RETT])
    
    return df_combined, df_RETT, df_CTRL


def correlation_matrix(df, titles, savename):
    # 计算相关性矩阵
    correlation_matrix = df.corr()

    # 设置图的尺寸
    plt.figure(figsize=(20, 16))

    # 使用Seaborn绘制热图
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')

    # 添加标题
    plt.title(f'Correlation Matrix of Features - {titles}')
#     plt.savefig(f'tables/{target}/{savename}_correlation_matrix.png', dpi=300)
    # 显示图表
    plt.show()



def validate_pca(df_combined, df_RETT, df_CTRL, target, savename):
    print("📊 PCA")
    # 提取特征数据
    features = df_combined.drop('State', axis=1)

    # 应用 PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    # 计算贡献率
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained variance ratio:", explained_variance_ratio)

    # 获取加载矩阵
    loading_matrix = pca.components_.T
    loading_df = pd.DataFrame(loading_matrix, columns=['PC1', 'PC2'], index=features.columns)

    # 可视化加载矩阵
    plt.figure(figsize=(16, 12))
    sns.heatmap(loading_df, annot=True, cmap='coolwarm')
    plt.title('PCA Loading Matrix')
#     plt.savefig(f'tables/{target}/{savename}_PCA_Matrix.png', dpi=300)
    plt.show()

    # 重置索引以确保对齐
    state_df = df_combined[['State']].reset_index(drop=True)
    finalDf = pd.concat([principalDf, state_df], axis=1)

    # 使用 Seaborn 绘制 PCA 结果图
    sns.scatterplot(data=finalDf, x='principal component 1', y='principal component 2', hue='State')
    plt.title('PCA of Dataset by State')
#     plt.savefig(f'tables/{target}/{savename}_PCA.png', dpi=300)
    plt.show()
    
    


def validate_ttest(df_combined, df_RETT, df_CTRL, target, savename):
    print("📊 ttest")
    # 初始化存储 p 值的列表
    p_values = []

    # 进行 t-检验
    for column in df_CTRL.columns[:-1]:  # 忽略 'State' 列
        t_stat, p_val = ttest_ind(df_CTRL[column], df_RETT[column], equal_var=False)  # 可以假设不等方差
        p_values.append((column, p_val))

    # 将 p 值转化为 DataFrame
    p_values_df = pd.DataFrame(p_values, columns=['Feature', 'p_value'])

    # 提取原始 p 值列表
    p_values_list = p_values_df['p_value'].tolist()

    # 进行校正
    rej, pval_corr = smm.multipletests(p_values_list, alpha=0.05, method='fdr_bh')[:2]

    # 将校正后的 p 值添加回 DataFrame
    p_values_df['p_corrected'] = pval_corr

    # 筛选显著特征（例如校正后 p < 0.02）
    significant_features = p_values_df[p_values_df['p_corrected'] < 0.02]

    # 按 p 值排序
    significant_features = significant_features.sort_values(by='p_corrected')
    significant_features = pd.DataFrame(significant_features)
    
    # 保存csv
    savepath = f"tables/{target}/{savename}_ttest.csv"
    significant_features.to_csv(savepath, index=False)
    print(f"Saved significant features to {savepath}")
    
#     # 打印 DataFrame
#     print("significant_features: ", len(significant_features))
#     print(significant_features.to_string(index=False))

#     # 可视化显著特征的 p 值
#     plt.figure(figsize=(16, 10))
#     sns.barplot(x='p_corrected', y='Feature', data=significant_features, palette='viridis')
#     plt.title('Significant Features Differentiating CTRL and RETT')
#     plt.xlabel('p_value_corrected')
#     plt.ylabel('Features')
#     plt.axvline(x=0.02, color='r', linestyle='--')
#     plt.savefig(f'tables/{target}/{savename}_ttest.png', dpi=300)
#     plt.show()
    
    return significant_features
    


def validate_mannwhitneyutest(df_combined, df_RETT, df_CTRL, target, savename):
    print("📊 Mann-Whitney U test")
    p_values = []
    # 进行 u-检验
    for column in df_CTRL.columns[:-1]:
        stat, p_val = mannwhitneyu(df_CTRL[column], df_RETT[column], alternative='two-sided')
        p_values.append((column, p_val))
    
    # 将 p 值转化为 DataFrame
    p_values_df = pd.DataFrame(p_values, columns=['Feature', 'p_value'])
    
    # 提取原始 p 值列表
    p_values_list = p_values_df['p_value'].tolist()
    
    # 提取原始 p 值列表
    rej, pval_corr = smm.multipletests(p_values_list, alpha=0.05, method='fdr_bh')[:2]
    
    # 将校正后的 p 值添加回 DataFrame
    p_values_df['p_corrected'] = pval_corr
    
    # 筛选显著特征（例如校正后 p < 0.02）
    significant_features = p_values_df[p_values_df['p_corrected'] < 0.02]
    
    # 按 p 值排序
    significant_features = significant_features.sort_values(by='p_corrected')
    significant_features = pd.DataFrame(significant_features)

    # 保存csv
    savepath = f"tables/{target}/{savename}_utest.csv"
    significant_features.to_csv(savepath, index=False)
    print(f"Saved significant features to {savepath}")
    
#     # 打印 DataFrame
#     print("significant_features: ", len(significant_features))
#     print(significant_features.to_string(index=False))

#     # 可视化显著特征的 p 值
#     plt.figure(figsize=(16, 10))
#     sns.barplot(x='p_corrected', y='Feature', data=significant_features, palette='viridis')
#     plt.title('Significant Features Differentiating CTRL and RETT')
#     plt.xlabel('p_value_corrected')
#     plt.ylabel('Features')
#     plt.axvline(x=0.02, color='r', linestyle='--')
#     plt.savefig(f'tables/{target}/{savename}_mannwhitneyutest.png', dpi=300)
#     plt.show()
    
    return significant_features
    
    
    
def validata_boxplot(data_all, target, rett_type, feature):
    # 假设 data_all 是之前整理好的 DataFrame
    unique_stains = data_all['Stain_Type'].unique()  # 获取所有染色类型
    p_values = []
    # 计算每种染色类型的 p 值
    for stain in unique_stains:
        group_ctrl = data_all[(data_all['State'] == 'CTRL') & (data_all['Stain_Type'] == stain)][feature]
        group_rett = data_all[(data_all['State'] == 'RETT') & (data_all['Stain_Type'] == stain)][feature]
        t_val, p_val = ttest_ind(group_ctrl, group_rett, equal_var=False)  # 可以假设不等方差
        # 单边检验：假设组1（RETT）的均值大于组2（CTRL）
        p_val = p_val / 2
        # 如果 t 值为负，则取 1 - (p 值 / 2)
        p_val = np.where(t_val < 0, 1 - p_val, p_val)
        p_values.append(p_val)
    for i in range(len(unique_stains)):
        print(f"p-value {unique_stains[i]}: {p_values[i]}")

    # 设置颜色
    palette_colors = {"CTRL": sns.color_palette(palette='bwr')[0], 
                      "RETT": sns.color_palette(palette='Pastel1')[0]}  # CTRL 使用绿色，RETT 使用紫色
    savepath = f'tables/{target}/{target}_{rett_type}_{feature}.png'
    # 绘制箱型图
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Stain_Type', y=feature, hue='State', data=data_all, palette=palette_colors)
    plt.title(f'{feature} with Stains and Cell States')
    plt.ylabel(f'{feature}')
    plt.xlabel('Stain Type')
    plt.legend(title='Cell State')
    plt.savefig(savepath, dpi=300)
    plt.show()
    print(f"Saved BOX plot to {savepath}")

            
def get_image_path(ctrl_type, chip_type, rett_type="HPS9999"):
    if ctrl_type=="RETT":
        image_path = f"{ctrl_type}_{rett_type}_{chip_type}"
    elif ctrl_type=="CTRL":
        image_path = f"{ctrl_type}_{chip_type}"
    return image_path