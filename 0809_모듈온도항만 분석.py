import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------
# 파일 경로 및 설정
# -----------------------
bms_file = r"E:\EV6_1year\59\bms_merged_59.csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

# -----------------------
# BMS 데이터 불러오기
# -----------------------
bms_df = pd.read_csv(bms_file)

def parse_mod_temp_list(val):
    try:
        nums = list(map(float, str(val).split(',')))
        return np.mean(nums)
    except:
        return np.nan

bms_df['mod_temp_avg'] = bms_df['mod_temp_list'].apply(parse_mod_temp_list)
bms_df['time'] = pd.to_datetime(bms_df['time'], errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)
bms_df['year_month'] = bms_df['time'].dt.to_period('M').dt.to_timestamp()
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)

# 세그먼트 구분
gap_indices = bms_df.index[bms_df['delta_sec_raw'] > 60].tolist()
segment_id = 0
segments = []
for i in range(len(bms_df)):
    segments.append(segment_id)
    if i in gap_indices:
        segment_id += 1
bms_df['segment_id'] = segments

# delta_sec 재계산
bms_df['delta_sec'] = bms_df.groupby('segment_id')['time'].diff().dt.total_seconds().fillna(0)
bms_df['valid'] = bms_df['delta_sec'] <= 10

# -----------------------
# OCV-SOC 보간 함수 정의
# -----------------------
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]]
soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float)

soc_vals = soc_ocv_data['SOC'].values / 100
ocv_vals = soc_ocv_data['OCV'].values
valid_mask = (~np.isnan(soc_vals)) & (~np.isnan(ocv_vals))
soc_vals = soc_vals[valid_mask]
ocv_vals = ocv_vals[valid_mask]
sort_idx = np.argsort(soc_vals)
soc_vals = soc_vals[sort_idx]
ocv_vals = ocv_vals[sort_idx]
ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value='extrapolate')

Qmax = 56.47

def compute_estored(soc):
    if soc <= 0.001:
        return 0
    result, _ = quad(ocv_func, 0, soc, limit=200)
    return Qmax * result * 192 * 2  # Wh

# -----------------------
# 월별 효율 + X1, X2 계산
# -----------------------
monthly_result = []

for month, df in bms_df.groupby('year_month'):
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    cable_col = 'chrg_cable_conn'
    speed_col = 'speed'
    current_col = 'pack_current'
    voltage_col = 'pack_volt'
    time_interval = 'delta_sec'
    soc_col = 'soc'

    def calc_energy(v, c, t):
        return (v * c * t) / 3600

    # 조건 정의
    cond_drive = df['valid'] & (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0)
    cond_idle = df['valid'] & (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0)
    cond_regen = df['valid'] & (df[cable_col] == 0) & (df[current_col] < 0)
    cond_chg = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] < 0)
    cond_chg_idle = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0)

    # X1: 충전조건 평균 온도
    X1 = df.loc[(df[cable_col] == 1), 'mod_temp_avg'].mean()

    # X2: 주행조건 평균 온도
    cond_rest = (df[cable_col] == 0) & (df[speed_col] == 0) & (df['delta_sec'] > 600)
    cond_drive_section = ~( (df[cable_col] == 1) | cond_rest )
    X2 = df.loc[cond_drive_section, 'mod_temp_avg'].mean()

    # 에너지 계산
    E_drive = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_drive].sum()
    E_idle = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_idle].sum()
    E_discharge = E_drive + E_idle
    E_regen = calc_energy(df[voltage_col], -df[current_col], df[time_interval])[cond_regen].sum()
    E_trip_net = E_discharge - E_regen
    E_chg_real = calc_energy(df[voltage_col], -df[current_col], df[time_interval])[cond_chg].sum()
    E_chg_idle = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_chg_idle].sum()
    E_charging = E_chg_real - E_chg_idle

    soc_series = df[df['valid']][soc_col].dropna()
    if len(soc_series) < 2:
        continue
    soc_0 = soc_series.iloc[0] / 100
    soc_1 = soc_series.iloc[-1] / 100
    E_stored_0 = compute_estored(soc_0)
    E_stored_1 = compute_estored(soc_1)
    E_stored_diff = E_stored_0 - E_stored_1

    # e_charge 계산
    df['chg_segment'] = (cond_chg != cond_chg.shift()).cumsum()
    df.loc[~cond_chg, 'chg_segment'] = np.nan
    chg_groups = df[df['chg_segment'].notna()].groupby('chg_segment')
    chg_e_stored_total = 0
    chg_energy_total = 0
    for _, g in chg_groups:
        s0, s1 = g[soc_col].iloc[0] / 100, g[soc_col].iloc[-1] / 100
        e0, e1 = compute_estored(s0), compute_estored(s1)
        e_chg = (g[voltage_col] * g[current_col].abs() * g[time_interval] / 3600).sum()
        chg_e_stored_total += max(e1 - e0, 0)
        chg_energy_total += e_chg
    if chg_energy_total > 0:
        e_chg_ratio = (chg_e_stored_total / chg_energy_total) * 100
    else:
        e_chg_ratio = np.nan

    # 효율 계산
    try:
        e1 = E_trip_net / (E_charging + E_stored_diff) * 100
        e2 = (E_trip_net + E_stored_1) / (E_charging + E_stored_0) * 100
    except ZeroDivisionError:
        e1, e2 = np.nan, np.nan

    monthly_result.append({
        'month': month,
        'efficiency1': e1,
        'efficiency2': e2,
        'X1': X1,
        'X2': X2
    })

# -----------------------
# 회귀분석
# -----------------------
df_result = pd.DataFrame(monthly_result).dropna()

X = df_result[['X1', 'X2']]

# Y1 모델
model_Y1 = LinearRegression()
model_Y1.fit(X, df_result['efficiency1'])
print("Y1 회귀계수:", model_Y1.intercept_, model_Y1.coef_)

# Y2 모델
model_Y2 = LinearRegression()
model_Y2.fit(X, df_result['efficiency2'])
print("Y2 회귀계수:", model_Y2.intercept_, model_Y2.coef_)

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯용

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------
# (위에서 작성한 데이터 처리 및 monthly_result 계산 부분 동일)
# -----------------------------------

df_result = pd.DataFrame(monthly_result).dropna()
X = df_result[['X1', 'X2']]

# Y1 모델
model_Y1 = LinearRegression()
model_Y1.fit(X, df_result['efficiency1'])
df_result['Y1_pred'] = model_Y1.predict(X)
print("Y1 회귀계수:", model_Y1.intercept_, model_Y1.coef_)

# Y2 모델
model_Y2 = LinearRegression()
model_Y2.fit(X, df_result['efficiency2'])
df_result['Y2_pred'] = model_Y2.predict(X)
print("Y2 회귀계수:", model_Y2.intercept_, model_Y2.coef_)

# -----------------------------------
# 시각화
# -----------------------------------

# 1. 3D 산점도 + 회귀면 (Y1)
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(df_result['X1'], df_result['X2'], df_result['efficiency1'], color='blue', label='실제 Y1')

# 회귀평면 생성
x1_range = np.linspace(df_result['X1'].min(), df_result['X1'].max(), 20)
x2_range = np.linspace(df_result['X2'].min(), df_result['X2'].max(), 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
y1_grid = model_Y1.intercept_ + model_Y1.coef_[0] * x1_grid + model_Y1.coef_[1] * x2_grid

ax.plot_surface(x1_grid, x2_grid, y1_grid, alpha=0.5, cmap='viridis')
ax.set_xlabel('X1 (충전 평균 온도)')
ax.set_ylabel('X2 (주행 평균 온도)')
ax.set_zlabel('효율 Y1')
ax.set_title('Y1 다중회귀 결과')

# 2. 3D 산점도 + 회귀면 (Y2)
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df_result['X1'], df_result['X2'], df_result['efficiency2'], color='red', label='실제 Y2')

y2_grid = model_Y2.intercept_ + model_Y2.coef_[0] * x1_grid + model_Y2.coef_[1] * x2_grid
ax2.plot_surface(x1_grid, x2_grid, y2_grid, alpha=0.5, cmap='plasma')
ax2.set_xlabel('X1 (충전 평균 온도)')
ax2.set_ylabel('X2 (주행 평균 온도)')
ax2.set_zlabel('효율 Y2')
ax2.set_title('Y2 다중회귀 결과')

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. 데이터 준비 (예시)
# ------------------------------
# X1, X2 -> 독립변수 / y -> 종속변수
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'X1': np.random.rand(n) * 10,
    'X2': np.random.rand(n) * 5,
    'efficiency1': np.random.rand(n) * 50
})

# ------------------------------
# 2. 회귀 분석
# ------------------------------
X = df[['X1', 'X2']]
X = sm.add_constant(X)  # 절편항 추가
y = df['efficiency1']

model = sm.OLS(y, X).fit()

# ------------------------------
# 3. 예측값 & 잔차
# ------------------------------
df['Y_pred'] = model.predict(X)
df['residual'] = y - df['Y_pred']

# ------------------------------
# 4. 평가 지표 계산
# ------------------------------
R2 = r2_score(y, df['Y_pred'])
RMSE = np.sqrt(mean_squared_error(y, df['Y_pred']))
beta = model.params
p_values = model.pvalues

# VIF 계산
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# ------------------------------
# 5. 결과 출력
# ------------------------------
print("📊 회귀 분석 결과")
print("-" * 40)
print(f"R²       : {R2:.4f}")
print(f"RMSE     : {RMSE:.4f}")
print("\nβ 계수")
print(beta)
print("\nP-value")
print(p_values)
print("\nVIF")
print(vif_data)

# ------------------------------
# 6. 시각화
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (1) 예측 vs 실제값
sns.scatterplot(x=df['Y_pred'], y=y, ax=axes[0])
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axes[0].set_xlabel("예측값 (Predicted)")
axes[0].set_ylabel("실제값 (Actual)")
axes[0].set_title("예측값 vs 실제값")

# (2) 잔차 플롯
sns.residplot(x=df['Y_pred'], y=df['residual'], lowess=True, ax=axes[1],
              scatter_kws={'alpha':0.7})
axes[1].set_xlabel("예측값 (Predicted)")
axes[1].set_ylabel("잔차 (Residuals)")
axes[1].set_title("잔차 플롯")

plt.tight_layout()
plt.show()







# -----------------------------------
# 예측 vs 실제 비교 + 잔차 플롯
# -----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Y1 예측 vs 실제
sns.scatterplot(x='efficiency1', y='Y1_pred', data=df_result, ax=axes[0, 0])
axes[0, 0].set_xlabel('실제 Y1')
axes[0, 0].set_ylabel('예측 Y1')
axes[0, 0].set_title('Y1 실제 vs 예측')

# Y1 Residual plot
sns.residplot(x='Y1_pred', y=df_result['efficiency1'] - df_result['Y1_pred'], lowess=True, ax=axes[0, 1])
axes[0, 1].set_xlabel('예측 Y1')
axes[0, 1].set_ylabel('잔차')
axes[0, 1].set_title('Y1 잔차')

# Y2 예측 vs 실제
sns.scatterplot(x='efficiency2', y='Y2_pred', data=df_result, ax=axes[1, 0], color='red')
axes[1, 0].set_xlabel('실제 Y2')
axes[1, 0].set_ylabel('예측 Y2')
axes[1, 0].set_title('Y2 실제 vs 예측')

# Y2 Residual plot
sns.residplot(x='Y2_pred', y=df_result['efficiency2'] - df_result['Y2_pred'], lowess=True, ax=axes[1, 1], color='red')
axes[1, 1].set_xlabel('예측 Y2')
axes[1, 1].set_ylabel('잔차')
axes[1, 1].set_title('Y2 잔차')

plt.tight_layout()
plt.show()

###################뻄###

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------
# (위에서 작성한 데이터 처리 및 monthly_result 계산 부분 동일)
# -----------------------------------

df_result = pd.DataFrame(monthly_result).dropna()
X = df_result[['X1', 'X2']]

# ===================================
# Y1 모델
# ===================================
model_Y1 = LinearRegression()
model_Y1.fit(X, df_result['efficiency1'])
df_result['Y1_pred'] = model_Y1.predict(X)

R2_Y1 = r2_score(df_result['efficiency1'], df_result['Y1_pred'])
RMSE_Y1 = mean_squared_error(df_result['efficiency1'], df_result['Y1_pred'], squared=False)

# statsmodels로 분석
X_const = sm.add_constant(X)
sm_model_Y1 = sm.OLS(df_result['efficiency1'], X_const).fit()

# VIF 계산
vif_data = pd.DataFrame()
vif_data["변수"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print("===== Y1 회귀 결과 =====")
print(f"beta_0: {model_Y1.intercept_:.4f}")
print(f"beta_1: {model_Y1.coef_[0]:.4f}")
print(f"beta_2: {model_Y1.coef_[1]:.4f}")
print(f"R²: {R2_Y1:.4f}")
print(f"Adj. R²: {sm_model_Y1.rsquared_adj:.4f}")
print(f"RMSE: {RMSE_Y1:.4f}")
print("\n--- VIF (다중공선성) ---")
print(vif_data)
print("\n--- Statsmodels Summary ---")
print(sm_model_Y1.summary())

# ===================================
# Y2 모델
# ===================================
model_Y2 = LinearRegression()
model_Y2.fit(X, df_result['efficiency2'])
df_result['Y2_pred'] = model_Y2.predict(X)

R2_Y2 = r2_score(df_result['efficiency2'], df_result['Y2_pred'])
RMSE_Y2 = mean_squared_error(df_result['efficiency2'], df_result['Y2_pred'], squared=False)

sm_model_Y2 = sm.OLS(df_result['efficiency2'], X_const).fit()

vif_data_Y2 = pd.DataFrame()
vif_data_Y2["변수"] = X_const.columns
vif_data_Y2["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print("\n===== Y2 회귀 결과 =====")
print(f"beta_0: {model_Y2.intercept_:.4f}")
print(f"beta_1: {model_Y2.coef_[0]:.4f}")
print(f"beta_2: {model_Y2.coef_[1]:.4f}")
print(f"R²: {R2_Y2:.4f}")
print(f"Adj. R²: {sm_model_Y2.rsquared_adj:.4f}")
print(f"RMSE: {RMSE_Y2:.4f}")
print("\n--- VIF (다중공선성) ---")
print(vif_data_Y2)
print("\n--- Statsmodels Summary ---")
print(sm_model_Y2.summary())

# ===================================
# 시각화
# ===================================

# 1. 3D 회귀면 시각화
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(df_result['X1'], df_result['X2'], df_result['efficiency1'], color='blue', label='실제 Y1')

x1_range = np.linspace(df_result['X1'].min(), df_result['X1'].max(), 20)
x2_range = np.linspace(df_result['X2'].min(), df_result['X2'].max(), 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
y1_grid = model_Y1.intercept_ + model_Y1.coef_[0] * x1_grid + model_Y1.coef_[1] * x2_grid

ax.plot_surface(x1_grid, x2_grid, y1_grid, alpha=0.5, cmap='viridis')
ax.set_xlabel('X1 (충전 평균 온도)')
ax.set_ylabel('X2 (주행 평균 온도)')
ax.set_zlabel('효율 Y1')
ax.set_title('Y1 다중회귀 결과')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df_result['X1'], df_result['X2'], df_result['efficiency2'], color='red', label='실제 Y2')

y2_grid = model_Y2.intercept_ + model_Y2.coef_[0] * x1_grid + model_Y2.coef_[1] * x2_grid
ax2.plot_surface(x1_grid, x2_grid, y2_grid, alpha=0.5, cmap='plasma')
ax2.set_xlabel('X1 (충전 평균 온도)')
ax2.set_ylabel('X2 (주행 평균 온도)')
ax2.set_zlabel('효율 Y2')
ax2.set_title('Y2 다중회귀 결과')

plt.tight_layout()
plt.show()

# 2. 예측 vs 실제 + 잔차
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.scatterplot(x='efficiency1', y='Y1_pred', data=df_result, ax=axes[0, 0])
axes[0, 0].set_xlabel('실제 Y1')
axes[0, 0].set_ylabel('예측 Y1')
axes[0, 0].set_title('Y1 실제 vs 예측')

sns.residplot(x='Y1_pred', y=df_result['efficiency1'] - df_result['Y1_pred'], lowess=True, ax=axes[0, 1])
axes[0, 1].set_xlabel('예측 Y1')
axes[0, 1].set_ylabel('잔차')
axes[0, 1].set_title('Y1 잔차')

sns.scatterplot(x='efficiency2', y='Y2_pred', data=df_result, ax=axes[1, 0], color='red')
axes[1, 0].set_xlabel('실제 Y2')
axes[1, 0].set_ylabel('예측 Y2')
axes[1, 0].set_title('Y2 실제 vs 예측')

sns.residplot(x='Y2_pred', y=df_result['efficiency2'] - df_result['Y2_pred'], lowess=True, ax=axes[1, 1], color='red')
axes[1, 1].set_xlabel('예측 Y2')
axes[1, 1].set_ylabel('잔차')
axes[1, 1].set_title('Y2 잔차')

plt.tight_layout()
plt.show()
