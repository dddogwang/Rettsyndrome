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


def loadcsv(target, stain_type, rett_type):
    if target=="features_image":
        loadpath_RETT = f"tables/{target}/features_{rett_type}_RETT_{stain_type}.csv"
        loadpath_CTRL = f"tables/{target}/features_HPS9999_CTRL_{stain_type}.csv"
    elif target=="features_ScoreCAM":
        loadpath_RETT = f"tables/{target}/features_{rett_type}_RETT_{stain_type}_Resnet10_noavg_ScoreCAM.csv"
        loadpath_CTRL = f"tables/{target}/features_{rett_type}_Ctrl_{stain_type}_Resnet10_noavg_ScoreCAM.csv"
    else:
        print(f"Load Failed, can not find {loadpath}")
    # 定义你想要读取的列的索引，注意 Python 索引从 0 开始
    columns_to_use = [10] + list(range(12, 19)) + list(range(20, 22)) + list(range(36, 99))
    # 读取 CSV 文件时仅加载指定的列
    df_RETT = pd.read_csv(loadpath_RETT, usecols=columns_to_use).dropna()  # 删除包含 NaN 的样本
    df_CTRL = pd.read_csv(loadpath_CTRL, usecols=columns_to_use).dropna()  # 删除包含 NaN 的样本
    print(f"🦠 LOAD {loadpath_RETT} {df_RETT.shape}")
    print(f"🧫 LOAD {loadpath_CTRL} {df_CTRL.shape}")
    
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

def loadcsv_Standard(target, stain_type, rett_type):
    if target=="features_image":
        loadpath_RETT = f"tables/{target}/features_{rett_type}_RETT_{stain_type}.csv"
        loadpath_CTRL = f"tables/{target}/features_HPS9999_CTRL_{stain_type}.csv"
    elif target=="features_ScoreCAM":
        loadpath_RETT = f"tables/{target}/features_{rett_type}_RETT_{stain_type}_Resnet10_noavg_ScoreCAM.csv"
        loadpath_CTRL = f"tables/{target}/features_{rett_type}_Ctrl_{stain_type}_Resnet10_noavg_ScoreCAM.csv"
    else:
        print(f"Load Failed, can not find {loadpath}")
        
    # 定义你想要读取的列的索引，注意 Python 索引从 0 开始
    columns_to_use = [10] + list(range(12, 19)) + list(range(20, 22)) + list(range(36, 99))
    # 读取 CSV 文件时仅加载指定的列
    df_RETT = pd.read_csv(loadpath_RETT, usecols=columns_to_use).dropna()  # 删除包含 NaN 的样本
    df_CTRL = pd.read_csv(loadpath_CTRL, usecols=columns_to_use).dropna()  # 删除包含 NaN 的样本
    print(f"🦠 LOAD {loadpath_RETT} {df_RETT.shape}")
    print(f"🧫 LOAD {loadpath_CTRL} {df_CTRL.shape}")
    
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



def validate_pca(df_combined, df_RETT, df_CTRL, loadpath, savename):
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
    plt.savefig(f'tables/{target}/{savename}_PCA_Matrix.png', dpi=300)
    plt.show()

    # 重置索引以确保对齐
    state_df = df_combined[['State']].reset_index(drop=True)
    finalDf = pd.concat([principalDf, state_df], axis=1)

    # 使用 Seaborn 绘制 PCA 结果图
    sns.scatterplot(data=finalDf, x='principal component 1', y='principal component 2', hue='State')
    plt.title('PCA of Dataset by State')
    plt.savefig(f'tables/{target}/{savename}_PCA.png', dpi=300)
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
        _, p_val = ttest_ind(group_ctrl, group_rett)
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

from skimage import feature, transform
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed

def is_close(point, other_points, threshold=5):
    """检查是否有任何点在阈值范围内"""
    for other in other_points:
        if np.linalg.norm(np.array(point) - np.array(other)) <= threshold:
            return True
    return False

def filter_contours_by_proximity(contours, nuclear_contours, proximity=5):
    """过滤掉靠近核心轮廓的轮廓"""
    new_contours = []
    # 展平 nuclear_contours 中的所有点
    all_nuclear_points = [point for contour in nuclear_contours for point in contour]

    for contour in contours:
        # 检查轮廓中的任何点是否靠近核心轮廓的点
        if not any(is_close(point, all_nuclear_points, proximity) for point in contour):
            new_contours.append(contour)

    return new_contours

def compute_largest_eigenvalue(image, sigma=1):
    nuclear = (image!=0).astype(np.uint8)
    nuclear_scaled = transform.rescale(nuclear, 48/50)
    nuclear_padded = np.pad(nuclear_scaled, pad_width=10, mode='constant', constant_values=0)

    # 计算结构张量
    result = feature.structure_tensor(image, sigma=sigma, order='rc')
    # 从结构张量中获取特征值
    eigenvalues = feature.structure_tensor_eigenvalues(result)
    # 返回每个点的最大特征值
    return np.max(eigenvalues, axis=0)*nuclear_padded

# 使用distance_transform_edt
def apply_h_watershed(image, min_distance=5):
    mask = image > threshold_otsu(image)
    # 计算距离变换
    distance = distance_transform_edt(mask)
    # 在距离图中找到峰值
    local_maxi = feature.peak_local_max(distance, min_distance=min_distance, labels=mask)
    # 将峰值的坐标转换为标记矩阵
    if len(local_maxi)<=255:
        markers = np.zeros_like(image, dtype=np.uint8)
    else:
#         print("len(local_maxi)>255")
        markers = np.zeros_like(image, dtype=np.int32)
    for i, (row, col) in enumerate(local_maxi):
        markers[row, col] = i + 1
    # 执行分水岭分割
    labels_ws = watershed(-distance, markers, mask=mask)
    return labels_ws


from skimage import measure

def calculate_quantitative_metrics(nucleus_image, cc_mask):
    """
    计算细胞核图像的量化指标。
    
    参数:
    nucleus_image: numpy.ndarray, 细胞核图像，灰度图
    cc_mask: numpy.ndarray, 染色中心的掩膜，二值图
    
    返回:
    metrics: dict, 包含所有量化指标的字典
    """
    metrics = {}
    
    # 计算可见染色中心的数量
    cc_labels = measure.label(cc_mask, connectivity=2)
    num_cc = np.max(cc_labels)
    metrics['chromatin_num'] = num_cc
    
    # 计算细胞核面积
    nuclear_area = np.sum(nucleus_image > 0)
    metrics['nuclear_area'] = nuclear_area
    
    # 计算平均chromatin面积 (CA)
    cc_areas = [np.sum(cc_labels == i) for i in range(1, num_cc + 1)]
#     metrics['relative_cc_areas'] = relative_cc_areas
    metrics['chromatin_area'] = np.mean(cc_areas)

    # 计算相对(核)chromatin面积和 (RCA-S)
    metrics['RCA-S'] = np.sum(cc_areas)/nuclear_area

    # 计算相对(核)chromatin面积平均 (RCA-M)
    metrics['RCA-M'] = np.mean(cc_areas)/nuclear_area
    
    # 计算细胞核强度平均
    nuclear_intensity = np.mean(nucleus_image[nucleus_image > 0])
    metrics['nuclear_intensity'] = nuclear_intensity

    # 计算平均chromatin强度平均 (CI-M)
    cc_intensities = [np.mean(nucleus_image[cc_labels == i]) for i in range(1, num_cc + 1)]
    metrics['chromatin_intensity'] = np.mean(cc_intensities)

    # 计算相对(核)chromatin强度和 (RCI-S)
    metrics['RCI-S'] = np.sum(cc_intensities)/nuclear_intensity

    # 计算相对(核)chromatin强度平均 (RCI-M)
    metrics['RCI-M'] = np.mean(cc_intensities)/nuclear_intensity
    
    # # 计算相对(核)chromatin比例 (RHF)
    # rhf = hf * rhi
    # metrics['relative_heterochromatin_fraction'] = rhf
    
    return metrics
 