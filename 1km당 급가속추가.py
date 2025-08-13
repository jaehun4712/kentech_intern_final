import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------
# 파일 경로 및 설정
# -----------------------
file_info = {
    r"E:\EV6_1year\06\bms_merged_06.csv": "veh06",
    r"E:\EV6_1year\49\bms_merged_49.csv": "veh49"
}

dfs = []
for path, vehicle_name in file_info.items():
    df_tmp = pd.read_csv(path)
    df_tmp['vehicle'] = vehicle_name
    dfs.append(df_tmp)

bms_df = pd.concat(dfs, ignore_index=True)

ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"


# -----------------------
# BMS 데이터 불러오기
# -----------------------

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

for (vehicle, month), df in bms_df.groupby(['vehicle', 'year_month']):
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    cable_col = 'chrg_cable_conn'
    speed_col = 'speed'
    current_col = 'pack_current'
    voltage_col = 'pack_volt'
    time_interval = 'delta_sec'
    soc_col = 'soc'

    # 새로운 독립변수 정의 (X3, X4)
    # 가속도(acceleration)는 기존 데이터에 있다고 가정
    # 급가속 횟수 (acceleration > 2m/s^2)
    X3 = df.loc[df['acceleration'] > 2, 'acceleration'].count()

    # 총 이동거리 (odometer의 차이)
    if 'odometer' in df.columns and len(df['odometer'].dropna()) > 1:
        X4 = df['odometer'].iloc[-1] - df['odometer'].iloc[0]
    else:
        X4 = np.nan

    # X3와 X4를 통합한 새로운 변수 X3_new
    if X4 > 0:
        X3_new = X3 / X4
    else:
        X3_new = np.nan


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
    cond_drive_section = ~((df[cable_col] == 1) | cond_rest)
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
        'vehicle': vehicle,
        'month': month,
        'efficiency1': e1,
        'efficiency2': e2,
        'X1': X1,
        'X2': X2,
        'X3_new': X3_new  # 새로운 독립변수 추가
    })

# -----------------------
# 회귀분석 및 시각화 (수정된 코드 시작)
# -----------------------
df_result = pd.DataFrame(monthly_result).dropna()

# 변수 중심화(Centering)를 적용
X1_mean = df_result['X1'].mean()
X2_mean = df_result['X2'].mean()

X_poly = pd.DataFrame()
X_poly['X1_centered'] = df_result['X1'] - X1_mean
X_poly['X2_centered'] = df_result['X2'] - X2_mean
X_poly['X1_sq_centered'] = X_poly['X1_centered'] ** 2
X_poly['X2_sq_centered'] = X_poly['X2_centered'] ** 2

# X3_new는 1차항으로만 추가하므로 중심화하지 않음
X_poly['X3_new'] = df_result['X3_new']

# statsmodels를 사용하여 절편항 추가
X_poly_const = sm.add_constant(X_poly)

# ===================================
# Y1 모델
# ===================================
y1 = df_result['efficiency1']
sm_model_Y1 = sm.OLS(y1, X_poly_const).fit()
df_result['Y1_pred'] = sm_model_Y1.predict(X_poly_const)

# VIF 계산 (이제 X_poly_const를 사용하여 올바르게 계산됩니다)
vif_data_Y1 = pd.DataFrame()
vif_data_Y1["변수"] = X_poly_const.columns
vif_data_Y1["VIF"] = [variance_inflation_factor(X_poly_const.values, i) for i in range(X_poly_const.shape[1])]

print("===== Y1 회귀 결과 =====")
print(sm_model_Y1.summary())
print("\n--- VIF (다중공선성) ---")
print(vif_data_Y1)

# ===================================
# Y2 모델
# ===================================
y2 = df_result['efficiency2']
sm_model_Y2 = sm.OLS(y2, X_poly_const).fit()
df_result['Y2_pred'] = sm_model_Y2.predict(X_poly_const)

# VIF 계산 (이제 X_poly_const를 사용하여 올바르게 계산됩니다)
vif_data_Y2 = pd.DataFrame()
vif_data_Y2["변수"] = X_poly_const.columns
vif_data_Y2["VIF"] = [variance_inflation_factor(X_poly_const.values, i) for i in range(X_poly_const.shape[1])]

print("\n===== Y2 회귀 결과 =====")
print(sm_model_Y2.summary())
print("\n--- VIF (다중공선성) ---")
print(vif_data_Y2)

# ===================================
# 시각화
# ===================================

# 1. 3D 회귀면 시각화 (X1, X2만 사용)
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(df_result['X1'], df_result['X2'], df_result['efficiency1'], color='blue', label='실제 Y1')

x1_range = np.linspace(df_result['X1'].min(), df_result['X1'].max(), 20)
x2_range = np.linspace(df_result['X2'].min(), df_result['X2'].max(), 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Y1 회귀면 계산 (다항식)
# X3_new의 평균값을 사용하여 3D 평면을 그림
x3_new_mean = df_result['X3_new'].mean()

y1_grid = sm_model_Y1.params['const'] + \
          sm_model_Y1.params['X1_centered'] * (x1_grid - X1_mean) + \
          sm_model_Y1.params['X2_centered'] * (x2_grid - X2_mean) + \
          sm_model_Y1.params['X1_sq_centered'] * ((x1_grid - X1_mean) ** 2) + \
          sm_model_Y1.params['X2_sq_centered'] * ((x2_grid - X2_mean) ** 2) + \
          sm_model_Y1.params['X3_new'] * x3_new_mean

ax1.plot_surface(x1_grid, x2_grid, y1_grid, alpha=0.5, cmap='viridis')
ax1.set_xlabel('X1 (충전 평균 온도)')
ax1.set_ylabel('X2 (주행 평균 온도)')
ax1.set_zlabel('효율 Y1')
ax1.set_title('Y1 다항회귀 결과 (X3_new는 평균값 사용)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df_result['X1'], df_result['X2'], df_result['efficiency2'], color='red', label='실제 Y2')

# Y2 회귀면 계산 (다항식)
y2_grid = sm_model_Y2.params['const'] + \
          sm_model_Y2.params['X1_centered'] * (x1_grid - X1_mean) + \
          sm_model_Y2.params['X2_centered'] * (x2_grid - X2_mean) + \
          sm_model_Y2.params['X1_sq_centered'] * ((x1_grid - X1_mean) ** 2) + \
          sm_model_Y2.params['X2_sq_centered'] * ((x2_grid - X2_mean) ** 2) + \
          sm_model_Y2.params['X3_new'] * x3_new_mean

ax2.plot_surface(x1_grid, x2_grid, y2_grid, alpha=0.5, cmap='plasma')
ax2.set_xlabel('X1 (충전 평균 온도)')
ax2.set_ylabel('X2 (주행 평균 온도)')
ax2.set_zlabel('효율 Y2')
ax2.set_title('Y2 다항회귀 결과 (X3_new는 평균값 사용)')

plt.tight_layout()
plt.show()

# 2. 예측 vs 실제 + 잔차
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.scatterplot(x=y1, y=df_result['Y1_pred'], data=df_result, ax=axes[0, 0])
axes[0, 0].set_xlabel('실제 Y1')
axes[0, 0].set_ylabel('예측 Y1')
axes[0, 0].set_title('Y1 실제 vs 예측')

sns.residplot(x=df_result['Y1_pred'], y=y1 - df_result['Y1_pred'], lowess=True, ax=axes[0, 1])
axes[0, 1].set_xlabel('예측 Y1')
axes[0, 1].set_ylabel('잔차')
axes[0, 1].set_title('Y1 잔차')

sns.scatterplot(x=y2, y=df_result['Y2_pred'], data=df_result, ax=axes[1, 0], color='red')
axes[1, 0].set_xlabel('실제 Y2')
axes[1, 0].set_ylabel('예측 Y2')
axes[1, 0].set_title('Y2 실제 vs 예측')

sns.residplot(x=df_result['Y2_pred'], y=y2 - df_result['Y2_pred'], lowess=True, ax=axes[1, 1], color='red')
axes[1, 1].set_xlabel('예측 Y2')
axes[1, 1].set_ylabel('잔차')
axes[1, 1].set_title('Y2 잔차')

plt.tight_layout()
plt.show()