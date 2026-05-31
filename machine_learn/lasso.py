import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. 制造带噪音和冗余特征的数据
# ==========================================
# 制造500个样本，20个特征。但实际上只有 5 个特征是真正有用的 (n_informative=5)
X, y, true_coef = make_regression(n_samples=500, n_features=20, n_informative=5, noise=15, coef=True, random_state=0)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. 核心预处理：数据标准化 (Z-score 归一化)
# ==========================================
# Lasso 对特征的尺度非常敏感，必须保证所有特征在同一量纲下
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 注意：测试集只能 transform，不能 fit，防止数据泄露

# ==========================================
# 3. 基础版：手动指定正则化强度 alpha 的 Lasso
# ==========================================
# alpha (即理论公式中的 lambda) 控制惩罚力度。alpha 越大，被压缩为0的特征越多。
lasso_manual = Lasso(alpha=2.0, max_iter=10000, tol=1e-4, random_state=42)
lasso_manual.fit(X_train_scaled, y_train)

# ==========================================
# 4. 进阶版 (工业界首选)：LassoCV 自动寻找最优 alpha
# ==========================================
# LassoCV 会自动在指定的 alpha 路径上进行 K 折交叉验证，找出泛化能力最好的 alpha
# cv=5 表示 5折交叉验证
lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, max_iter=10000, tol=1e-4, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

best_alpha = lasso_cv.alpha_
print(f"LassoCV 自动寻找到的最优 alpha 值为: {best_alpha:.4f}\n")

# ==========================================
# 5. 模型评估与特征稀疏性观察
# ==========================================
# 使用最优模型在测试集上预测
y_pred = lasso_cv.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- 模型评估指标 ---")
print(f"测试集 MSE: {mse:.4f}")
print(f"测试集 R-squared: {r2:.4f}\n")

print("--- 特征选择能力观察 (Lasso 的核心) ---")
# 提取模型训练后的特征权重
learned_coefs = lasso_cv.coef_

# 统计被剔除的特征数量 (权重被压缩为绝对的 0)
eliminated_features = np.sum(learned_coefs == 0)
retained_features = np.sum(learned_coefs != 0)

print(f"总特征数: {X.shape[1]}")
print(f"被 Lasso 压缩为 0 的特征数: {eliminated_features} (这些是模型认定为噪音/冗余的特征)")
print(f"被 Lasso 保留的特征数: {retained_features}")

# 对比真实有用的特征（我们设定的是5个）
print(f"数据生成时设定的真实有效特征数: 5")
