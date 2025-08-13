import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 파일 경로 및 설정
# -----------------------
file_paths = [
    r"E:\EV6_1year\59\bms_merged_59.csv",
    r"E:\EV6_1year\57\bms_merged_57.csv"
]
bms_df = pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)


ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------
# BMS 데이터 불러오기
# -----------------------
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
# 월별 효율 계산
# -----------------------
monthly_result = []
for month, month_df in bms_df.groupby('year_month'):
    df = month_df.copy()
    df.columns = [col.lower() for col in df.columns]

    cable_col = 'chrg_cable_conn'
    speed_col = 'speed'
    current_col = 'pack_current'
    voltage_col = 'pack_volt'
    time_interval = 'delta_sec'
    soc_col = 'soc'


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
    soc_0 = soc_series.iloc[0] / 100
    soc_1 = soc_series.iloc[-1] / 100
    E_stored_0 = compute_estored(soc_0)
    E_stored_1 = compute_estored(soc_1)
    E_stored_diff = E_stored_0 - E_stored_1

    # e_charge 계산 (세그먼트 기반)
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

    avg_temp = df['ext_temp'].mean()
    monthly_result.append(
        {'month': month, 'efficiency1': e1, 'efficiency2': e2, 'e_charge': e_chg_ratio, 'ext_temp': avg_temp})

# -----------------------
# 시각화 및 상관관계 분석
# -----------------------
monthly_df = pd.DataFrame(monthly_result).dropna()

# 상관계수
corr_matrix = monthly_df[['efficiency1', 'efficiency2', 'e_charge', 'ext_temp']].corr()
print("\n📊 상관계수 행렬:")
print(corr_matrix)

# 시각화
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
sns.lineplot(data=monthly_df, x='month', y='ext_temp', ax=axs[0], color='tab:blue')
axs[0].set_title('월별 평균 외기온도')
axs[0].set_ylabel('외기온도 (°C)')

sns.lineplot(data=monthly_df, x='month', y='efficiency1', ax=axs[1], label='efficiency1')
sns.lineplot(data=monthly_df, x='month', y='efficiency2', ax=axs[1], label='efficiency2')
axs[1].set_title('월별 주행 효율 (e1, e2)')
axs[1].set_ylabel('효율 (%)')

sns.lineplot(data=monthly_df, x='month', y='e_charge', ax=axs[2], label='e_charge', color='tab:red')
axs[2].set_title('월별 충전 효율 (e_charge)')
axs[2].set_ylabel('충전 효율 (%)')
axs[2].set_xlabel('월')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

months = monthly_df['month'].dt.month
labels = monthly_df['month'].dt.strftime('%Y-%m')

fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

# 그래프 1: 외기온도 + efficiency1
axs[0].plot(months, monthly_df['ext_temp'], color='tab:blue', marker='o', label='외기온도 (°C)')
axs[0].plot(months, monthly_df['efficiency1'], color='tab:green', marker='s', label='주행 효율 e1 (%)')
for x, y1, y2 in zip(months, monthly_df['ext_temp'], monthly_df['efficiency1']):
    axs[0].text(x, y1 + 0.5, f'{y1:.1f}', color='tab:blue', ha='center', fontsize=9)
    axs[0].text(x, y2 + 0.5, f'{y2:.1f}', color='tab:green', ha='center', fontsize=9)

axs[0].set_title('월별 외기온도 & 주행 효율 e1')
axs[0].set_ylabel('온도 / 효율 (%)')
axs[0].legend()
axs[0].grid(True)

# 그래프 2: 외기온도 + efficiency2
axs[1].plot(months, monthly_df['ext_temp'], color='tab:blue', marker='o', label='외기온도 (°C)')
axs[1].plot(months, monthly_df['efficiency2'], color='tab:orange', marker='s', label='주행 효율 e2 (%)')
for x, y1, y2 in zip(months, monthly_df['ext_temp'], monthly_df['efficiency2']):
    axs[1].text(x, y1 + 0.5, f'{y1:.1f}', color='tab:blue', ha='center', fontsize=9)
    axs[1].text(x, y2 + 0.5, f'{y2:.1f}', color='tab:orange', ha='center', fontsize=9)

axs[1].set_title('월별 외기온도 & 주행 효율 e2')
axs[1].set_ylabel('온도 / 효율 (%)')
axs[1].legend()
axs[1].grid(True)

# 그래프 3: 외기온도 + e_charge
axs[2].plot(months, monthly_df['ext_temp'], color='tab:blue', marker='o', label='외기온도 (°C)')
axs[2].plot(months, monthly_df['e_charge'], color='tab:red', marker='s', label='충전 효율 e_charge (%)')
for x, y1, y2 in zip(months, monthly_df['ext_temp'], monthly_df['e_charge']):
    axs[2].text(x, y1 + 0.5, f'{y1:.1f}', color='tab:blue', ha='center', fontsize=9)
    axs[2].text(x, y2 + 0.5, f'{y2:.1f}', color='tab:red', ha='center', fontsize=9)

axs[2].set_title('월별 외기온도 & 충전 효율 e_charge')
axs[2].set_ylabel('온도 / 효율 (%)')
axs[2].legend()
axs[2].grid(True)

# x축 설정
axs[2].set_xticks(range(1, 13))
axs[2].set_xticklabels([f'{i}월' for i in range(1, 13)])
axs[2].set_xlabel('월')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 상관계수 및 유의확률 계산
r1, p1 = pearsonr(monthly_df['ext_temp'], monthly_df['efficiency1'])
r2, p2 = pearsonr(monthly_df['ext_temp'], monthly_df['efficiency2'])
r3, p3 = pearsonr(monthly_df['ext_temp'], monthly_df['e_charge'])

months = monthly_df['month'].dt.month

fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

# 그래프 1: 외기온도 + efficiency1
axs[0].plot(months, monthly_df['ext_temp'], color='tab:blue', marker='o', label='외기온도 (°C)')
axs[0].plot(months, monthly_df['efficiency1'], color='tab:green', marker='s', label='주행 효율 e1 (%)')
for x, y1, y2 in zip(months, monthly_df['ext_temp'], monthly_df['efficiency1']):
    axs[0].text(x, y1 + 0.5, f'{y1:.1f}', color='tab:blue', ha='center', fontsize=9)
    axs[0].text(x, y2 + 0.5, f'{y2:.1f}', color='tab:green', ha='center', fontsize=9)
axs[0].text(0.02, 0.95, f'r = {r1:.2f}, p = {p1:.3f}', transform=axs[0].transAxes,
            fontsize=12, color='black', verticalalignment='top')
axs[0].set_title('월별 외기온도 & 주행 효율 e1')
axs[0].set_ylabel('온도 / 효율 (%)')
axs[0].legend()
axs[0].grid(True)

# 그래프 2: 외기온도 + efficiency2
axs[1].plot(months, monthly_df['ext_temp'], color='tab:blue', marker='o', label='외기온도 (°C)')
axs[1].plot(months, monthly_df['efficiency2'], color='tab:orange', marker='s', label='주행 효율 e2 (%)')
for x, y1, y2 in zip(months, monthly_df['ext_temp'], monthly_df['efficiency2']):
    axs[1].text(x, y1 + 0.5, f'{y1:.1f}', color='tab:blue', ha='center', fontsize=9)
    axs[1].text(x, y2 + 0.5, f'{y2:.1f}', color='tab:orange', ha='center', fontsize=9)
axs[1].text(0.02, 0.95, f'r = {r2:.2f}, p = {p2:.3f}', transform=axs[1].transAxes,
            fontsize=12, color='black', verticalalignment='top')
axs[1].set_title('월별 외기온도 & 주행 효율 e2')
axs[1].set_ylabel('온도 / 효율 (%)')
axs[1].legend()
axs[1].grid(True)

# 그래프 3: 외기온도 + e_charge
axs[2].plot(months, monthly_df['ext_temp'], color='tab:blue', marker='o', label='외기온도 (°C)')
axs[2].plot(months, monthly_df['e_charge'], color='tab:red', marker='s', label='충전 효율 e_charge (%)')
for x, y1, y2 in zip(months, monthly_df['ext_temp'], monthly_df['e_charge']):
    axs[2].text(x, y1 + 0.5, f'{y1:.1f}', color='tab:blue', ha='center', fontsize=9)
    axs[2].text(x, y2 + 0.5, f'{y2:.1f}', color='tab:red', ha='center', fontsize=9)
axs[2].text(0.02, 0.95, f'r = {r3:.2f}, p = {p3:.3f}', transform=axs[2].transAxes,
            fontsize=12, color='black', verticalalignment='top')
axs[2].set_title('월별 외기온도 & 충전 효율 e_charge')
axs[2].set_ylabel('온도 / 효율 (%)')
axs[2].legend()
axs[2].grid(True)

# x축: 1~12월 전부 표시
axs[2].set_xticks(range(1, 13))
axs[2].set_xticklabels([f'{i}월' for i in range(1, 13)])
axs[2].set_xlabel('월')

plt.tight_layout()
plt.show()
