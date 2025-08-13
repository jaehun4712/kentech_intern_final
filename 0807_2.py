import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 파일 경로 및 설정
# -----------------------

# -----------------------
# BMS 데이터 불러오기
# -----------------------
file_paths = [

r"E:\EV6_1year\51\bms_merged_51.csv",
r"E:\EV6_1year\53\bms_merged_53.csv"

]
bms_df = pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)


#bms_file = r"E:\EV6_1year\51\bms_merged_51.csv"
#bms_df = pd.read_csv(bms_file)




ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False



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

# 세그먼트 구분
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
# 주 단위 효율 계산
# -----------------------
weekly_result = []
for week, df in bms_df.groupby('week'):
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

    try:
        e1 = E_trip_net / (E_charging + E_stored_diff) * 100
        e2 = (E_trip_net + E_stored_1) / (E_charging + E_stored_0) * 100
    except ZeroDivisionError:
        e1, e2 = np.nan, np.nan

    avg_temp = df['mod_temp_avg'].mean()

    weekly_result.append({
        'week': week,
        'efficiency1': e1,
        'efficiency2': e2,
        'mod_temp_avg': avg_temp
    })

# -----------------------
# 결과 DataFrame 및 시각화
# -----------------------
weekly_df = pd.DataFrame(weekly_result).dropna()

# 상관계수
r1, p1 = pearsonr(weekly_df['mod_temp_avg'], weekly_df['efficiency1'])
r2, p2 = pearsonr(weekly_df['mod_temp_avg'], weekly_df['efficiency2'])

# 선 그래프 (1주 단위 추세)
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
sns.lineplot(data=weekly_df, x='week', y='mod_temp_avg', ax=axs[0], color='tab:blue')
axs[0].set_title('주간 평균 모듈온도')
axs[0].set_ylabel('모듈온도')

sns.lineplot(data=weekly_df, x='week', y='efficiency1', ax=axs[1], label='efficiency1')
sns.lineplot(data=weekly_df, x='week', y='efficiency2', ax=axs[1], label='efficiency2')
axs[1].set_title('주간 주행 효율 (e1, e2)')
axs[1].set_ylabel('효율 (%)')

#sns.lineplot(data=weekly_df, x='week', y='e_charge', ax=axs[2], label='e_charge', color='tab:red')
#axs[2].set_title('주간 충전 효율 (e_charge)')
#axs[2].set_ylabel('충전 효율 (%)')
#axs[2].set_xlabel('주')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 산점도 (온도 vs 효율)
fig, axs = plt.subplots(2, 1, figsize=(10, 14), sharex=False)

axs[0].scatter(weekly_df['mod_temp_avg'], weekly_df['efficiency1'], color='tab:green')
axs[0].set_title(f'모듈온도 vs 주행 효율 e1 (r={r1:.2f}, p={p1:.3f})')
axs[0].set_xlabel('모듈온도 (°C)')
axs[0].set_ylabel('주행 효율 e1 (%)')
axs[0].grid(True)

axs[1].scatter(weekly_df['mod_temp_avg'], weekly_df['efficiency2'], color='tab:orange')
axs[1].set_title(f'모듈온도 vs 주행 효율 e2 (r={r2:.2f}, p={p2:.3f})')
axs[1].set_xlabel('모듈온도 (°C)')
axs[1].set_ylabel('주행 효율 e2 (%)')
axs[1].grid(True)

plt.tight_layout()
plt.show()

######추가
# 회귀선 포함 산점도
fig, axs = plt.subplots(2, 1, figsize=(10, 14), sharex=False)
sns.regplot(data=weekly_df, x='mod_temp_avg', y='efficiency1', ax=axs[0], scatter=True, ci=95, color='tab:green')
axs[0].set_title(f'온도 vs 효율 e1 (r={r1:.2f}, p={p1:.3f})')

sns.regplot(data=weekly_df, x='mod_temp_avg', y='efficiency2', ax=axs[1], scatter=True, ci=95, color='tab:orange')
axs[1].set_title(f'온도 vs 효율 e2 (r={r2:.2f}, p={p2:.3f})')

for ax in axs:
    ax.set_xlabel('모듈온도(°C)')
    ax.set_ylabel('효율(%)')
    ax.grid(True)

plt.tight_layout()
plt.show()


# 추세 그래프
plt.figure(figsize=(14, 5))
sns.lineplot(data=weekly_df, x='week', y='efficiency1', label='efficiency1')
sns.lineplot(data=weekly_df, x='week', y='efficiency2', label='efficiency2')
sns.lineplot(data=weekly_df, x='week', y='mod_temp_avg', label='mod_temp_avg', color='tab:blue')
plt.title('주간 효율 및 모듈온도 추세')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 산점도 + 회귀선
fig, axs = plt.subplots(3, 1, figsize=(10, 14), sharex=False)

sns.regplot(data=weekly_df, x='mod_temp_avg', y='efficiency1', ax=axs[0], scatter=True, ci=95, color='tab:green')
axs[0].set_title(f'온도 vs 효율 e1 (r={r1:.2f}, p={p1:.3f})')

sns.regplot(data=weekly_df, x='mod_temp_avg', y='efficiency2', ax=axs[1], scatter=True, ci=95, color='tab:orange')
axs[1].set_title(f'온도 vs 효율 e2 (r={r2:.2f}, p={p2:.3f})')


for ax in axs:
    ax.set_xlabel('모듈온도 (°C)')
    ax.set_ylabel('효율 (%)')
    ax.grid(True)

plt.tight_layout()
plt.show()