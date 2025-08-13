import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# -----------------------
# 데이터 로딩 (차량 구분 추가)
# -----------------------
bms_df_59 = pd.read_csv(r"E:\EV6_1year\59\bms_merged_59.csv")
bms_df_59["vehicle"] = "veh59"

bms_df_57 = pd.read_csv(r"E:\EV6_1year\57\bms_merged_57.csv")
bms_df_57["vehicle"] = "veh57"

bms_df = pd.concat([bms_df_59, bms_df_57], ignore_index=True)

# -----------------------
# 기본 전처리
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

bms_df['week'] = bms_df['time'].dt.to_period('W').dt.start_time
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)

gap_indices = bms_df.index[bms_df['delta_sec_raw'] > 60].tolist()
segment_id = 0
segments = []
for i in range(len(bms_df)):
    segments.append(segment_id)
    if i in gap_indices:
        segment_id += 1
bms_df['segment_id'] = segments
bms_df['delta_sec'] = bms_df.groupby('segment_id')['time'].diff().dt.total_seconds().fillna(0)
bms_df['valid'] = bms_df['delta_sec'] <= 10

# -----------------------
# OCV-SOC 보간
# -----------------------
ocv_raw = pd.read_excel(r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx", sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]].dropna().astype(float)
soc_ocv_data.columns = ['SOC', 'OCV']

soc_vals = soc_ocv_data['SOC'].values / 100
ocv_vals = soc_ocv_data['OCV'].values
valid_mask = (~np.isnan(soc_vals)) & (~np.isnan(ocv_vals))
soc_vals = soc_vals[valid_mask]
ocv_vals = ocv_vals[valid_mask]
sort_idx = np.argsort(soc_vals)
soc_vals, ocv_vals = soc_vals[sort_idx], ocv_vals[sort_idx]

ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value='extrapolate')
Qmax = 56.47

def compute_estored(soc):
    if soc <= 0.001:
        return 0
    result, _ = quad(ocv_func, 0, soc, limit=200)
    return Qmax * result * 192 * 2

# -----------------------
# 차량별 주간 효율 계산
# -----------------------
weekly_result = []
for (vehicle, week), df in bms_df.groupby(['vehicle', 'week']):
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    cable_col, speed_col, current_col, voltage_col = 'chrg_cable_conn', 'speed', 'pack_current', 'pack_volt'
    time_interval, soc_col = 'delta_sec', 'soc'

    def calc_energy(v, c, t):
        return (v * c * t) / 3600

    cond_drive = df['valid'] & (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0)
    cond_idle = df['valid'] & (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0)
    cond_regen = df['valid'] & (df[cable_col] == 0) & (df[current_col] < 0)
    cond_chg = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] < 0)
    cond_chg_idle = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0)

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
    soc_0, soc_1 = soc_series.iloc[0] / 100, soc_series.iloc[-1] / 100
    E_stored_0, E_stored_1 = compute_estored(soc_0), compute_estored(soc_1)
    E_stored_diff = E_stored_0 - E_stored_1

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

    e_chg_ratio = (chg_e_stored_total / chg_energy_total) * 100 if chg_energy_total > 0 else np.nan

    try:
        e1 = E_trip_net / (E_charging + E_stored_diff) * 100
        e2 = (E_trip_net + E_stored_1) / (E_charging + E_stored_0) * 100
    except ZeroDivisionError:
        e1, e2 = np.nan, np.nan

    avg_temp = df['mod_temp_avg'].mean()

    weekly_result.append({
        'vehicle': vehicle,
        'week': week,
        'efficiency1': e1,
        'efficiency2': e2,
        'e_charge': e_chg_ratio,
        'mod_temp_avg': avg_temp
    })

weekly_df = pd.DataFrame(weekly_result).dropna()

# -----------------------
# 시각화
# -----------------------
plt.figure(figsize=(14, 6))
sns.lineplot(data=weekly_df, x='week', y='efficiency1', hue='vehicle')
plt.title("주간 Efficiency1 비교")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.lineplot(data=weekly_df, x='week', y='e_charge', hue='vehicle')
plt.title("주간 충전 효율 (e_charge) 비교")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.lineplot(data=weekly_df, x='week', y='mod_temp_avg', hue='vehicle')
plt.title("주간 평균 모듈온도 비교")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

from scipy.stats import linregress

# x, y 데이터 준비
x = weekly_df['mod_temp_avg']
y = weekly_df['efficiency1']

# 선형 회귀 분석
result = linregress(x, y)

# 회귀선 그리기
plt.figure(figsize=(8,6))
sns.scatterplot(x=x, y=y, color='tab:green', label='Data Points')
sns.lineplot(x=x, y=result.intercept + result.slope * x, color='tab:blue', label='Fit Line')

# 통계값 텍스트 추가
r_value = result.rvalue
p_value = result.pvalue
r_squared = r_value**2

plt.text(0.05, 0.95, f"r = {r_value:.3f}\np = {p_value:.3e}\nR² = {r_squared:.3f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

plt.xlabel('모듈온도(°C)')
plt.ylabel('효율 e1 (%)')
plt.title('모듈온도 vs 효율 e1 (전체 차량 데이터)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# efficiency2 데이터
x2 = weekly_df['mod_temp_avg']
y2 = weekly_df['efficiency2']

# 선형 회귀 분석
result2 = linregress(x2, y2)

# 그래프 그리기
plt.figure(figsize=(8,6))
sns.scatterplot(x=x2, y=y2, color='tab:orange', label='Data Points')
sns.lineplot(x=x2, y=result2.intercept + result2.slope * x2, color='tab:red', label='Fit Line')

# 통계값 텍스트 추가
r_value2 = result2.rvalue
p_value2 = result2.pvalue
r_squared2 = r_value2**2

plt.text(0.05, 0.95, f"r = {r_value2:.3f}\np = {p_value2:.3e}\nR² = {r_squared2:.3f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

plt.xlabel('모듈온도(°C)')
plt.ylabel('효율 e2 (%)')
plt.title('모듈온도 vs 효율 e2 (전체 차량 데이터)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
