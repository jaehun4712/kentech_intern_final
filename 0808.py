import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 파일 불러오기
# -----------------------
file_paths = [
    r"E:\EV6_1year\51\bms_merged_51.csv",
    r"E:\EV6_1year\53\bms_merged_53.csv"
]
bms_df = pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)

ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------
# 모듈온도 평균 계산
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
# OCV-SOC lookup table 생성 (빠른 E_stored 계산)
# -----------------------
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]]
soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float)

soc_vals = soc_ocv_data['SOC'].values / 100
ocv_vals = soc_ocv_data['OCV'].values
order = np.argsort(soc_vals)
soc_vals = soc_vals[order]
ocv_vals = ocv_vals[order]

Qmax = 56.47
# SOC 구간을 세밀하게 나눠서 미리 적분
soc_grid = np.linspace(0, 1, 1001)
ocv_interp = interp1d(soc_vals, ocv_vals, kind='linear', fill_value='extrapolate')(soc_grid)
estored_grid = Qmax * cumulative_trapezoid(ocv_interp, soc_grid, initial=0) * 192 * 2  # Wh

def fast_estored(soc):
    return np.interp(soc, soc_grid, estored_grid)

# -----------------------
# 에너지 계산 (벡터화)
# -----------------------
calc_energy = lambda v, c, t: (v * c * t) / 3600

cond_drive = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 0) & (bms_df['speed'] > 0) & (bms_df['pack_current'] > 0)
cond_idle = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 0) & (bms_df['speed'] == 0) & (bms_df['pack_current'] > 0)
cond_regen = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 0) & (bms_df['pack_current'] < 0)
cond_chg = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 1) & (bms_df['speed'] == 0) & (bms_df['pack_current'] < 0)
cond_chg_idle = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 1) & (bms_df['speed'] == 0) & (bms_df['pack_current'] > 0)

bms_df['E_drive'] = calc_energy(bms_df['pack_volt'], bms_df['pack_current'], bms_df['delta_sec']) * cond_drive
bms_df['E_idle'] = calc_energy(bms_df['pack_volt'], bms_df['pack_current'], bms_df['delta_sec']) * cond_idle
bms_df['E_regen'] = calc_energy(bms_df['pack_volt'], -bms_df['pack_current'], bms_df['delta_sec']) * cond_regen
bms_df['E_chg_real'] = calc_energy(bms_df['pack_volt'], -bms_df['pack_current'], bms_df['delta_sec']) * cond_chg
bms_df['E_chg_idle'] = calc_energy(bms_df['pack_volt'], bms_df['pack_current'], bms_df['delta_sec']) * cond_chg_idle

# -----------------------
# 주차별 SOC → E_stored 계산
# -----------------------
def soc_diff_energy(df):
    soc_series = df[df['valid']]['soc'].dropna()
    if len(soc_series) < 2:
        return np.nan, np.nan, np.nan
    soc0 = soc_series.iloc[0] / 100
    soc1 = soc_series.iloc[-1] / 100
    return fast_estored(soc0), fast_estored(soc1), fast_estored(soc0) - fast_estored(soc1)

weekly_df = bms_df.groupby('week').apply(
    lambda g: pd.Series({
        'E_drive': g['E_drive'].sum(),
        'E_idle': g['E_idle'].sum(),
        'E_regen': g['E_regen'].sum(),
        'E_chg_real': g['E_chg_real'].sum(),
        'E_chg_idle': g['E_chg_idle'].sum(),
        'E_stored_0': soc_diff_energy(g)[0],
        'E_stored_1': soc_diff_energy(g)[1],
        'E_stored_diff': soc_diff_energy(g)[2],
        'mod_temp_avg': g['mod_temp_avg'].mean()
    })
).reset_index()

# -----------------------
# 효율 계산
# -----------------------
E_trip_net = (weekly_df['E_drive'] + weekly_df['E_idle']) - weekly_df['E_regen']
E_charging = weekly_df['E_chg_real'] - weekly_df['E_chg_idle']

weekly_df['efficiency1'] = E_trip_net / (E_charging + weekly_df['E_stored_diff']) * 100
weekly_df['efficiency2'] = (E_trip_net + weekly_df['E_stored_1']) / (E_charging + weekly_df['E_stored_0']) * 100

weekly_df = weekly_df.dropna()

# -----------------------
# 상관계수
# -----------------------
r1, p1 = pearsonr(weekly_df['mod_temp_avg'], weekly_df['efficiency1'])
r2, p2 = pearsonr(weekly_df['mod_temp_avg'], weekly_df['efficiency2'])

# -----------------------
# 시각화
# -----------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
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
