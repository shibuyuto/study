import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

print("Anaconda test")

# データ読み込み
df = pd.read_csv("C:\\Users\\Yuto Shibuya\\Downloads\\default of credit card clients.csv")

# 1. 必要な変数を抽出
columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'default payment next month']
data = df[columns]

# 2. カラム名の変更
print(data.info())
print(data.describe())

# 2. 債権不履行の有無の度数とヒストグラム
print(data['default payment next month'].value_counts())

# 棒グラフを作成
sns.histplot(data['default payment next month'], bins=2, kde=False, color='skyblue', edgecolor='black')

# タイトルと軸ラベルの設定
plt.title('Default Payment Next Month', fontsize=16, fontweight='bold')
plt.xlabel('Default Payment', fontsize=12)
plt.ylabel('Count', fontsize=12)

# 軸の範囲を設定して見やすくする
plt.xticks([0, 1])  # 0=No Default, 1=Default

# グラフを表示
plt.show()

plt.hist(data['default payment next month'], bins=2, color='skyblue', edgecolor='black')
plt.xticks([0, 1]) 
plt.gca().spines['bottom'].set_position(('data', 0))  # X軸を0の位置に配置
plt.show()

df['default payment next month'].value_counts().plot.bar()

df['default payment next month'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])

# 3. 性別、学歴、婚姻状況の度数
print(data['SEX'].value_counts())
print(data['EDUCATION'].value_counts())
print(data['MARRIAGE'].value_counts())

# 4. クレジット上限、年齢の要約統計量と箱ひげ図
print(data[['LIMIT_BAL', 'AGE']].describe())
sns.boxplot(data=data[['LIMIT_BAL', 'AGE']])
plt.title('Boxplot of LIMIT_BAL and AGE')
plt.show()

# データ加工
# 1. 学歴と婚姻状況の欠損値処理
data['EDUCATION'] = data['EDUCATION'].replace({5: 5, 6: 5, 4: np.nan, 0: np.nan})
data['MARRIAGE'] = data['MARRIAGE'].replace({0: np.nan})

# 2. 欠損値を含むレコードを削除
data = data.dropna()

# 3. ワンホットエンコーディング
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_columns = encoder.fit_transform(data[['SEX', 'EDUCATION', 'MARRIAGE']])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['SEX', 'EDUCATION', 'MARRIAGE']))
data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1).drop(['SEX', 'EDUCATION', 'MARRIAGE'], axis=1)

# 4. 標準化
scaler = StandardScaler()
data[['LIMIT_BAL', 'AGE']] = scaler.fit_transform(data[['LIMIT_BAL', 'AGE']])

# 5. データ分割
seed = 42  # 学籍番号の下2桁を仮定
X = data.drop('default payment next month', axis=1)
y = data['default payment next month']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

# k-NNモデル構築と評価
# 1. k=7でモデル構築
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train[['LIMIT_BAL', 'AGE']], y_train)
y_pred_knn = knn.predict(X_test[['LIMIT_BAL', 'AGE']])
print(confusion_matrix(y_test, y_pred_knn))
print('Accuracy:', accuracy_score(y_test, y_pred_knn))
print('F1 Score:', f1_score(y_test, y_pred_knn))

# 2. クロスバリデーションで最適なkを探索
k_values = range(1, 22, 2)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train[['LIMIT_BAL', 'AGE']], y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
optimal_k = k_values[np.argmax(cv_scores)]
print('Optimal k:', optimal_k)

# 最適なkでモデル構築
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train[['LIMIT_BAL', 'AGE']], y_train)
y_pred_knn_opt = knn.predict(X_test[['LIMIT_BAL', 'AGE']])
print(confusion_matrix(y_test, y_pred_knn_opt))
print('Accuracy:', accuracy_score(y_test, y_pred_knn_opt))
print('F1 Score:', f1_score(y_test, y_pred_knn_opt))

# ロジスティック回帰
# 1. 性別のみを用いた解析
log_reg = LogisticRegression()
log_reg.fit(X_train[['SEX_2']], y_train)
print('Odds Ratio (Male vs Female):', np.exp(log_reg.coef_[0][0]))

# 2. クレジット上限のみを用いた解析
log_reg.fit(X_train[['LIMIT_BAL']], y_train)
print('Coefficient for LIMIT_BAL:', log_reg.coef_[0][0])

# ランダムフォレスト
# 1. モデル構築
rf = RandomForestClassifier(n_estimators=300, random_state=seed)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred_rf))
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print('F1 Score:', f1_score(y_test, y_pred_rf))

# 2. クロスバリデーションで最適なアンサンブル回数を探索
ensemble_sizes = [100, 200, 300, 400, 500]
cv_scores_rf = []
for size in ensemble_sizes:
    rf = RandomForestClassifier(n_estimators=size, random_state=seed)
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores_rf.append(scores.mean())
optimal_size = ensemble_sizes[np.argmax(cv_scores_rf)]
print('Optimal Ensemble Size:', optimal_size)

# 最適なアンサンブル回数でモデル構築
rf = RandomForestClassifier(n_estimators=optimal_size, random_state=seed)
rf.fit(X_train, y_train)
y_pred_rf_opt = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred_rf_opt))
print('Accuracy:', accuracy_score(y_test, y_pred_rf_opt))
print('F1 Score:', f1_score(y_test, y_pred_rf_opt))

