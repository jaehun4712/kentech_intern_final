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
    #r"E:\EV6_1year\49\bms_merged_49.csv": "veh49",
    #r"E:\EV6_1year\51\bms_merged_51.csv": "veh51",
    #r"E:\EV6_1year\53\bms_merged_53.csv": "veh53"
#r"E:\EV6_1year\54\bms_merged_54.csv": "veh54",
#r"E:\EV6_1year\57\bms_merged_57.csv": "veh57",
#r"E:\EV6_1year\59\bms_merged_59.csv": "veh59",
#r"E:\EV6_1year\73\bms_merged_73.csv": "veh73",
#r"E:\EV6_1year\76\bms_merged_76.csv": "veh76",
#r"E:\EV6_1year\84\bms_merged_84.csv": "veh84",
#r"E:\EV6_1year\86\bms_merged_86.csv": "veh86",
#r"E:\EV6_1year\87\bms_merged_87.csv": "veh87",
#r"E:\EV6_1year\90\bms_merged_90.csv": "veh90",
#r"E:\EV6_1year\91\bms_merged_91.csv": "veh91",
#r"E:\EV6_1year\92\bms_merged_92.csv": "veh92",
#r"E:\EV6_1year\94\bms_merged_94.csv": "veh94",
#r"E:\EV6_1year\95\bms_merged_95.csv": "veh95",
#r"E:\EV6_1year\103\bms_merged_103.csv": "veh103",
#r"E:\EV6_1year\104\bms_merged_104.csv": "veh104",
#r"E:\EV6_1year\106\bms_merged_106.csv": "veh106",
#r"E:\EV6_1year\107\bms_merged_107.csv": "veh107",
#r"E:\EV6_1year\132\bms_merged_132.csv": "veh132"


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

    # 새로운 독립변수 정의
    # 급가속 횟수 (acceleration > 2m/s^2)
    X3 = df.loc[df['acceleration'] > 2, 'acceleration'].count()

    # 총 이동거리 (odometer의 차이)
    if 'odometer' in df.columns and len(df['odometer'].dropna()) > 1:
        X4_odometer = df['odometer'].iloc[-1] - df['odometer'].iloc[0]
    else:
        X4_odometer = np.nan

    # X3와 X4를 통합한 새로운 변수 X3_new (1km당 급가속 횟수)
    if X4_odometer > 0:
        X3_new = X3 / (X4_odometer / 1000)  # 주행거리를 km 단위로 변환하여 계산
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

    # 주행 시간 조건
    cond_rest = (df[cable_col] == 0) & (df[speed_col] == 0) & (df['delta_sec'] > 600)
    cond_drive_section = ~((df[cable_col] == 1) | cond_rest)

    # X1: 충전조건 평균 온도
    X1 = df.loc[(df[cable_col] == 1), 'mod_temp_avg'].mean()

    # X2: 주행조건 평균 온도
    X2 = df.loc[cond_drive_section, 'mod_temp_avg'].mean()

    # 누락되었던 X5_new: 한 달 주행 중 평균 속도
    total_driving_distance_m = df.loc[cond_drive_section, 'delta_dist'].sum() if 'delta_dist' in df.columns else np.nan
    total_driving_time_sec = df.loc[cond_drive_section, 'delta_sec'].sum()
    if total_driving_time_sec > 0 and total_driving_distance_m > 0:
        X5_new = (total_driving_distance_m / 1000) / (total_driving_time_sec / 3600)  # km/h
    else:
        X5_new = np.nan

    # 새로운 통합 독립변수 X_eff_time_new (총 에너지 비소모 주행 시간)
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
        'X5_new': X5_new,  # X5_new 추가
        'X_eff_time_new': X_eff_time_new
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

# X3_new, X5_new, X_eff_time_new는 1차항으로만 추가
X_poly['X3_new'] = df_result['X3_new']
X_poly['X5_new'] = df_result['X5_new']
X_poly['X_eff_time_new'] = df_result['X_eff_time_new']

# statsmodels를 사용하여 절편항 추가
X_poly_const = sm.add_constant(X_poly)

# ===================================
# Y1 모델
# ===================================
y1 = df_result['efficiency1']
sm_model_Y1 = sm.OLS(y1, X_poly_const).fit()
df_result['Y1_pred'] = sm_model_Y1.predict(X_poly_const)

# VIF 계산
vif_data_Y1 = pd.DataFrame()
vif_data_Y1["변수"] = X_poly_const.columns
vif_data