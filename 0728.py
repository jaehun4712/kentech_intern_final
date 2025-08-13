import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

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
# Trip 단위 e_trip 계산 예시
# ----------------------------
def calc_energy(voltage, current, time_s):
    return (voltage * current * time_s) / 3600

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
        print(f"[Trip {tid}] e_trip = {e_trip_eff:.2f}% ({e_trip_actual:.1f} / {e_stored_diff:.1f} Wh)")
    else:
        print(f"[Trip {tid}] SOC 변화 적음 → e_trip 계산 생략")

# ----------------------------
# 전체 e_trip 계산
# ----------------------------
total_e_trip_actual = 0
total_e_stored_diff = 0

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
        total_e_trip_actual += e_trip_actual
        total_e_stored_diff += e_stored_diff

# 최종 e_trip 계산
if total_e_stored_diff > 0:
    final_e_trip = (total_e_trip_actual / total_e_stored_diff) * 100
    print(f"\n🔹 전체 e_trip = {final_e_trip:.2f}%")
    print(f"    ⤷ total actual: {total_e_trip_actual:.1f} Wh")
    print(f"    ⤷ total stored : {total_e_stored_diff:.1f} Wh")
else:
    print("\n❗ 총 E_stored 차이가 0이어서 e_trip 계산 불가")



import matplotlib.pyplot as plt

import matplotlib
from matplotlib import rc

# 한글 폰트 설정 (Windows의 경우)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
matplotlib.rcParams['axes.unicode_minus'] = False     # 음수 기호 깨짐 방지

import matplotlib.pyplot as plt

trip_summary = []

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
    else:
        e_trip_eff = np.nan

    trip_summary.append({
        'trip_id': tid,
        'e_drive': e_drive_discharge,
        'e_idle': e_idle_discharge,
        'e_regen': e_regen_charge,
        'e_stored_diff': e_stored_diff,
        'e_trip_actual': e_trip_actual,
        'e_trip_eff': e_trip_eff
    })

# → DataFrame으로 변환
trip_summary_df = pd.DataFrame(trip_summary)

# ----------------------------
# 시각화: Stacked Bar Plot
# ----------------------------
fig, ax1 = plt.subplots(figsize=(15, 6))

x = trip_summary_df['trip_id']
bar1 = ax1.bar(x, trip_summary_df['e_drive'], label='Drive Discharge', color='red')
bar2 = ax1.bar(x, trip_summary_df['e_idle'], bottom=trip_summary_df['e_drive'], label='Idle Discharge', color='orange')
bar3 = ax1.bar(x, -trip_summary_df['e_regen'], label='Regen Charge', color='blue')

ax1.set_ylabel('Energy (Wh)')
ax1.set_xlabel('Trip ID')
ax1.set_title('Trip별 방전/회생 에너지 구성')
ax1.legend(loc='upper right')
ax1.grid(True)

# ----------------------------
# 시각화: 효율 라인 추가
# ----------------------------
ax2 = ax1.twinx()
ax2.plot(x, trip_summary_df['e_trip_eff'], 'g--o', label='e_trip (%)')
ax2.set_ylabel('Efficiency (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0, 120)

plt.tight_layout()
plt.show()
