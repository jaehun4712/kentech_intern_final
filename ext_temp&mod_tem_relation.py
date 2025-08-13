import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
# 파일 경로
file_path = r"E:\EV6_1year\59\bms_merged_59.csv"

# CSV 파일 불러오기
df = pd.read_csv(file_path)

# 'mod_temp_list'를 평균값으로 변환
def parse_mod_temp_list(val):
    try:
        nums = list(map(float, str(val).split(',')))
        return np.mean(nums)
    except:
        return np.nan

df['mod_temp_avg'] = df['mod_temp_list'].apply(parse_mod_temp_list)

# 필요한 열만 추출
cols = ['ext_temp', 'mod_temp_avg', 'soc']
df_corr = df[cols].dropna()

# 상관계수 계산
corr_matrix = df_corr.corr()

# 상관계수 출력
print("상관계수:")
print(corr_matrix)

# 시각화 (히트맵)
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("상관관계 (Correlation)")
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# 산점도 행렬 (pairplot)
sns.pairplot(df_corr)
plt.suptitle("산점도 행렬 (ext_temp, mod_temp_avg, soc)", y=1.02)
plt.show()

# 개별 산점도 + 회귀선 (예시)
sns.lmplot(data=df_corr, x='ext_temp', y='mod_temp_avg')
plt.title("외기온도 vs 모듈 온도 평균")
plt.show()


