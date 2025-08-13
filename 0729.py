import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib
from matplotlib import rc

# 한글 폰트 설정 (Windows의 경우)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
matplotlib.rcParams['axes.unicode_minus'] = False     # 음수 기호 깨짐 방지

import matplotlib.pyplot as plt

# ----------------------------
# 파일 경로
# ----------------------------
bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\3week_data\bms_01241225206_2023-08.csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

# ----------------------------
# BMS 데이터 로드 및 시간 처리
# ----------------------------
bms_df = pd.read_csv(bms_file)
bms_df['time'] = pd.to_datetime(bms_df['time'], format='mixed', errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)

# delta_sec 및 날짜 처리
bms_df['delta_sec'] = bms_df['time'].diff().dt.total_seconds().fillna(0)
bms_df['date'] = bms_df['time'].dt.date

# delta_sec 이상치 제거
bms_df['delta_sec'] = bms_df['delta_sec'].apply(lambda x: x if x <= 300 else 0)

# ----------------------------
# OCV 곡선 처리
# ----------------------------
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
ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value="extrapolate")

Qmax = 56.47
n_cells = 192
parallel = 2
Qmax_total = Qmax * n_cells * parallel

# ----------------------------
# E_stored 함수 정의
# ----------------------------
def compute_estored(soc):
    if isinstance(soc, (np.ndarray, list, pd.Series)):
        soc = np.clip(soc, 0.001, 0.999)
        return np.array([compute_estored(s) for s in soc])
    if soc <= 0.001:
        return 0
    try:
        result, _ = quad(ocv_func, 0, soc, limit=1000, epsabs=1e-6, epsrel=1e-6)
        return Qmax * result * n_cells * parallel
    except Exception as e:
        print(f"[적분 실패] SOC: {soc:.4f} → 오류: {e}")
        return 0

# ----------------------------
# Trip ID 부여
# ----------------------------
df = bms_df.copy()
df['trip_id'] = np.nan
df = df.sort_values('time').reset_index(drop=True)

trip_id = 0
in_trip = False
trip_ids = []

for i in range(len(df)):
    row = df.iloc[i]
    if i > 0:
        prev_row = df.iloc[i - 1]
        date_changed = row['date'] != prev_row['date']
    else:
        date_changed = False

    if not in_trip:
        if (row['speed'] > 0) and (row['chrg_cable_conn'] == 0):
            in_trip = True
            trip_start_idx = i
            trip_ids.append(trip_id)
        else:
            trip_ids.append(np.nan)
    else:
        is_charging = row['chrg_cable_conn'] == 1
        is_long_idle = (row['speed'] == 0) and (row['delta_sec'] > 300)
        is_date_change_stop = (row['speed'] == 0) and date_changed

        if is_charging or is_long_idle or is_date_change_stop:
            in_trip = False
            trip_ids.append(np.nan)
            trip_id += 1
        else:
            trip_ids.append(trip_id)

df['trip_id'] = trip_ids

# ----------------------------
# Trip 단위 e_trip 계산 및 시각화
# ----------------------------
def calc_energy(voltage, current, time_s):
    return (voltage * current * time_s) / 3600

trip_eff_list = []
trip_groups = df[df['trip_id'].notna()].groupby('trip_id')
for tid, trip_df in trip_groups:
    trip_df = trip_df.sort_values('time')
    soc_start = trip_df['soc'].iloc[0] / 100
    soc_end = trip_df['soc'].iloc[-1] / 100
    e_start = compute_estored(soc_start)
    e_end = compute_estored(soc_end)
    e_stored_diff = max(e_start - e_end, 0)

    cond_drive_discharge = (trip_df['chrg_cable_conn'] == 0) & (trip_df['speed'] > 0) & (trip_df['pack_current'] > 0)
    cond_idle_discharge = (trip_df['chrg_cable_conn'] == 0) & (trip_df['speed'] == 0) & (trip_df['pack_current'] > 0)
    cond_regen_charge = (trip_df['chrg_cable_conn'] == 0) & (trip_df['pack_current'] < 0)

    e_drive_discharge = calc_energy(trip_df['pack_volt'], trip_df['pack_current'], trip_df['delta_sec'])[cond_drive_discharge].sum()
    e_idle_discharge = calc_energy(trip_df['pack_volt'], trip_df['pack_current'], trip_df['delta_sec'])[cond_idle_discharge].sum()
    e_regen_charge = calc_energy(trip_df['pack_volt'], -trip_df['pack_current'], trip_df['delta_sec'])[cond_regen_charge].sum()
    e_trip_actual = e_drive_discharge + e_idle_discharge - e_regen_charge

    if e_stored_diff > 0:
        e_trip_eff = (e_trip_actual / e_stored_diff) * 100
        trip_eff_list.append((tid, e_trip_eff, e_trip_actual, e_stored_diff))
    else:
        trip_eff_list.append((tid, np.nan, e_trip_actual, e_stored_diff))

trip_eff_df = pd.DataFrame(trip_eff_list, columns=['trip_id', 'e_trip_eff', 'e_trip_actual', 'e_stored_diff'])

# trip별 효율 시각화
plt.figure(figsize=(12, 6))
plt.plot(trip_eff_df['trip_id'], trip_eff_df['e_trip_eff'], marker='o', linestyle='-', color='blue')
plt.title('Trip 별 e_trip 효율')
plt.xlabel('Trip ID')
plt.ylabel('Efficiency (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 전체 평균 효율
total_e_actual = trip_eff_df['e_trip_actual'].sum()
total_e_stored = trip_eff_df['e_stored_diff'].sum()
e_trip_total = (total_e_actual / total_e_stored) * 100
print(f"\n✅ 전체 e_trip = {e_trip_total:.2f}%")
print(f"    ⤷ total actual: {total_e_actual:,.1f} Wh")
print(f"    ⤷ total stored : {total_e_stored:,.1f} Wh")

# ----------------------------
# 전체 구간에서의 Power 분석 (trip 외 포함)
# ----------------------------
df['power'] = calc_energy(df['pack_volt'], df['pack_current'], df['delta_sec'])

plt.figure(figsize=(15, 4))
plt.plot(df['time'], df['power'], label='Power (Wh)', color='gray', linewidth=0.5)

trip_mask = df['trip_id'].notna()
plt.plot(df.loc[trip_mask, 'time'], df.loc[trip_mask, 'power'], 'r.', markersize=2, label='Trip 구간')
plt.title('전체 데이터에서 Power 흐름 (Trip 포함)')
plt.xlabel('Time')
plt.ylabel('Power (Wh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Trip 외 구간에서 전류, 속도, delta_sec 분포
# ----------------------------
non_trip = df[df['trip_id'].isna()]

fig, axs = plt.subplots(1, 3, figsize=(18, 4))
axs[0].hist(non_trip['pack_current'], bins=100, color='skyblue', edgecolor='k')
axs[0].set_title('Trip 외 전류 분포')
axs[0].set_xlabel('Current (A)')

axs[1].hist(non_trip['speed'], bins=50, color='lightgreen', edgecolor='k')
axs[1].set_title('Trip 외 속도 분포')
axs[1].set_xlabel('Speed (km/h)')

axs[2].hist(non_trip['delta_sec'], bins=50, color='salmon', edgecolor='k')
axs[2].set_title('Trip 외 delta_sec 분포')
axs[2].set_xlabel('delta_sec (s)')

plt.tight_layout()
plt.show()
