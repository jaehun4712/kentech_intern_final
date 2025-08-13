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
    r"E:\EV6_1year\49\bms_merged_49.csv": "veh49",
    #r"E:\EV6_1year\51\bms_merged_51.csv": "veh51",
    #r"E:\EV6_1year\53\bms_merged_53.csv": "veh53",
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
# 월별 효율 + 독립변수 계산
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

    # 새로운 독립변수
    # 급가속 횟수
    X3 = df.loc[df['acceleration'] > 2, 'acceleration'].count()

    # 총 이동거리
    if 'odometer' in df.columns and len(df['odometer'].dropna()) > 1:
        X4_odometer = df['odometer'].iloc[-1] - df['odometer'].iloc[0]
    else:
        X4_odometer = np.nan

    # 1km당 급가속 횟수
    if X4_odometer > 0:
        X3_new = X3 / (X4_odometer / 1000)
    else:
        X3_new = np.nan

    # 월 평균 속도 (km/h)
    if speed_col in df.columns:
        X_avg_speed = df[speed_col].mean()
    else:
        X_avg_speed = np.nan

    def calc_energy(v, c, t):
        return (v * c * t) / 3600

    # 조건 정의
    cond_drive = df['valid'] & (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0)
    cond_idle = df['valid'] & (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0)
    cond_regen = df['valid'] & (df[cable_col] == 0) & (df[current_col] < 0)
    cond_chg = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] < 0)
    cond_chg_idle = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0)

    # X1, X2
    X1 = df.loc[(df[cable_col] == 1), 'mod_temp_avg'].mean()
    cond_rest = (df[cable_col] == 0) & (df[speed_col] == 0) & (df['delta_sec'] > 600)
    cond_drive_section = ~((df[cable_col] == 1) | cond_rest)
    X2 = df.loc[cond_drive_section, 'mod_temp_avg'].mean()

    # 총 에너지 비소모 주행 시간
    cond_coasting = (df['speed'] > 0) & (df['chrg_cable_conn'] == 0) & (abs(df['pack_current']) < 1)
    cond_eff_driving = cond_regen | cond_coasting
    X_eff_time_new = df.loc[cond_eff_driving, time_interval].sum()

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
        'X3_new': X3_new,
        'X_eff_time_new': X_eff_time_new,
        'X_avg_speed': X_avg_speed
    })

# -----------------------
# 회귀분석 및 시각화
# -----------------------
df_result = pd.DataFrame(monthly_result).dropna()

# 변수 중심화
X1_mean = df_result['X1'].mean()
X2_mean = df_result['X2'].mean()

X_poly = pd.DataFrame()
X_poly['X1_centered'] = df_result['X1'] - X1_mean
X_poly['X2_centered'] = df_result['X2'] - X2_mean
X_poly['X1_sq_centered'] = X_poly['X1_centered'] ** 2
X_poly['X2_sq_centered'] = X_poly['X2_centered'] ** 2

# 나머지 변수들 (중심화X)
X_poly['X3_new'] = df_result['X3_new']
X_poly['X_eff_time_new'] = df_result['X_eff_time_new']
X_poly['X_avg_speed'] = df_result['X_avg_speed']

# statsmodels 절편항 추가
X_poly_const = sm.add_constant(X_poly)

# Y1 모델
y1 = df_result['efficiency1']
sm_model_Y1 = sm.OLS(y1, X_poly_const).fit()
df_result['Y1_pred'] = sm_model_Y1.predict(X_poly_const)

# VIF
vif_data_Y1 = pd.DataFrame()
vif_data_Y1["변수"] = X_poly_const.columns
vif_data_Y1["VIF"] = [variance_inflation_factor(X_poly_const.values, i) for i in range(X_poly_const.shape[1])]

print("===== Y1 회귀 결과 =====")
print(sm_model_Y1.summary())
print("\n--- VIF ---")
print(vif_data_Y1)

# Y2 모델
y2 = df_result['efficiency2']
sm_model_Y2 = sm.OLS(y2, X_poly_const).fit()
df_result['Y2_pred'] = sm_model_Y2.predict(X_poly_const)

# VIF
vif_data_Y2 = pd.DataFrame()
vif_data_Y2["변수"] = X_poly_const.columns
vif_data_Y2["VIF"] = [variance_inflation_factor(X_poly_const.values, i) for i in range(X_poly_const.shape[1])]

print("\n===== Y2 회귀 결과 =====")
print(sm_model_Y2.summary())
print("\n--- VIF ---")
print(vif_data_Y2)

# -----------------------
# 시각화
# -----------------------
independent_vars = ['X1', 'X2', 'X3_new', 'X_eff_time_new', 'X_avg_speed']
titles = {
    'X1': 'X1 (충전 평균 온도) - °C',
    'X2': 'X2 (주행 평균 온도) - °C',
    'X3_new': 'X3 (1km당 급가속 횟수) - 횟수/km',
    'X_eff_time_new': 'X (총 에너지 비소모 주행 시간) - 초',
    'X_avg_speed': 'X (월 평균 속도) - km/h'
}

# 각 변수와 Y1 관계
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.flatten()
fig.suptitle('독립변수와 효율 Y1', fontsize=16, y=1.02)

for i, var in enumerate(independent_vars):
    sns.scatterplot(x=df_result[var], y=df_result['efficiency1'], ax=axes[i], color='blue')
    sns.regplot(x=df_result[var], y=df_result['efficiency1'], ax=axes[i], scatter=False, color='red', line_kws={'linestyle': '--'})
    axes[i].set_title(f'{titles[var]}')
    axes[i].set_xlabel(f'{titles[var]}')
    axes[i].set_ylabel('효율 Y1')
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# 각 변수와 Y2 관계
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.flatten()
fig.suptitle('독립변수와 효율 Y2', fontsize=16, y=1.02)

for i, var in enumerate(independent_vars):
    sns.scatterplot(x=df_result[var], y=df_result['efficiency2'], ax=axes[i], color='blue')
    sns.regplot(x=df_result[var], y=df_result['efficiency2'], ax=axes[i], scatter=False, color='red', line_kws={'linestyle': '--'})
    axes[i].set_title(f'{titles[var]}')
    axes[i].set_xlabel(f'{titles[var]}')
    axes[i].set_ylabel('효율 Y2')
    axes[i].grid(True)

plt.tight_layout()
plt.show()
