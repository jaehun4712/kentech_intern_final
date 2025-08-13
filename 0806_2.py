import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
import ast

# ----------------------------
# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 파일 경로
bms_file = r"C:/Users/OWNER/Desktop/EV6_2301/EV6_1year/bms_01241225206_2023-08 (1).csv"
ocv_file = r"C:/Users/OWNER/Desktop/EV6_2301/NE_Cell_Characterization_performance.xlsx"

# ----------------------------
# BMS 데이터 로드 및 전처리
bms_df = pd.read_csv(bms_file)
bms_df['time'] = pd.to_datetime(bms_df['time'], errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)

# 평균 셀 전압 계산
def parse_and_avg_volts(volt_str):
    try:
        volt_list = ast.literal_eval(volt_str)
        return np.mean(volt_list)
    except:
        return np.nan

bms_df['cell_volt_avg'] = bms_df['cell_volt_list'].apply(parse_and_avg_volts)
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)
bms_df['delta_sec'] = bms_df['delta_sec_raw']
bms_df['date'] = bms_df['time'].dt.date
bms_df['delta_sec_power'] = bms_df['delta_sec_raw'].where(
    (bms_df['delta_sec_raw'] <= 300) & (bms_df['date'] == bms_df['date'].shift(1)), 0)

# ----------------------------
# OCV-SOC 보간 함수 생성
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]]
soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float)
soc_vals = soc_ocv_data['SOC'].values / 100
ocv_vals = soc_ocv_data['OCV'].values
valid = (~np.isnan(soc_vals)) & (~np.isnan(ocv_vals))
soc_vals, ocv_vals = soc_vals[valid], ocv_vals[valid]
sort_idx = np.argsort(soc_vals)
soc_vals, ocv_vals = soc_vals[sort_idx], ocv_vals[sort_idx]
ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value='extrapolate')
soc_func = interp1d(ocv_vals, soc_vals, kind='linear', fill_value='extrapolate')

Qmax = 56.47
n_cells = 192
parallel = 2

def compute_estored(soc):
    soc = np.clip(soc, 0.001, 0.999)
    try:
        result, _ = quad(ocv_func, 0, soc)
        return Qmax * result * n_cells * parallel
    except:
        return 0

# ----------------------------
# Trip 구간 설정 및 유효성 체크

df = bms_df.copy()
df['trip_id'] = np.nan
trip_id = 0
in_trip = False
i = 0

while i < len(df):
    if not in_trip:
        if df.loc[i, 'speed'] > 0 and df.loc[i, 'chrg_cable_conn'] == 0:
            trip_start = i
            in_trip = True
    else:
        if df.loc[i, 'speed'] == 0 and df.loc[i - 1, 'speed'] > 0:
            stop_idx = i
            j = i + 1
            while j < len(df) and df.loc[j, 'speed'] == 0:
                if df.loc[j, 'delta_sec'] > 300:
                    df.loc[trip_start:stop_idx - 1, 'trip_id'] = trip_id
                    trip_id += 1
                    in_trip = False
                    i = stop_idx - 1
                    break
                j += 1
        if df.loc[i, 'chrg_cable_conn'] == 1:
            df.loc[trip_start:i, 'trip_id'] = trip_id
            trip_id += 1
            in_trip = False
    i += 1

valid_trip_ids = []
trip_rest_volts = {}
trip_ids = df['trip_id'].dropna().unique()

for tid in trip_ids:
    trip_df = df[df['trip_id'] == tid]
    trip_start_idx = trip_df.index[0]
    trip_end_idx = trip_df.index[-1]

    if trip_start_idx == 0 or trip_end_idx >= len(df) - 1:
        continue

    # 시작 전 rest 조건
    prev_idx = trip_start_idx - 1
    while prev_idx >= 0 and df.loc[prev_idx, 'speed'] == 0 and df.loc[prev_idx, 'chrg_cable_conn'] == 0:
        if df.loc[prev_idx + 1, 'delta_sec'] > 7200:
            start_rest_idx = prev_idx
            break
        prev_idx -= 1
    else:
        continue

    # 종료 후 rest 조건
    next_idx = trip_end_idx + 1
    while next_idx < len(df) and df.loc[next_idx, 'speed'] == 0 and df.loc[next_idx, 'chrg_cable_conn'] == 0:
        if df.loc[next_idx, 'delta_sec'] > 7200:
            end_rest_idx = next_idx
            break
        next_idx += 1
    else:
        continue

    trip_rest_volts[tid] = {
        'start_volt': df.loc[start_rest_idx, 'cell_volt_avg'],
        'end_volt': df.loc[end_rest_idx, 'cell_volt_avg']
    }
    valid_trip_ids.append(tid)

df['trip_id'] = df['trip_id'].where(df['trip_id'].isin(valid_trip_ids))

# ----------------------------
# Trip 효율 계산 (전압 기반 SOC 추정)
def calc_energy(voltage, current, time_s):
    return (voltage * current * time_s) / 3600

trip_eff_list = []
trip_groups = df[df['trip_id'].notna()].groupby('trip_id')

for tid, trip_df in trip_groups:
    try:
        start_volt = trip_rest_volts[tid]['start_volt']
        end_volt = trip_rest_volts[tid]['end_volt']

        if np.isnan(start_volt) or np.isnan(end_volt):
            print(f"[무시됨] Trip {tid}: 전압 NaN 발생")
            continue

        soc_0 = soc_func(start_volt)
        soc_1 = soc_func(end_volt)
        e_start = compute_estored(soc_0)
        e_end = compute_estored(soc_1)
        e_stored_diff = max(e_start - e_end, 0)

        cond_drive = (trip_df['chrg_cable_conn'] == 0) & (trip_df['speed'] > 0) & (trip_df['pack_current'] > 0)
        cond_idle = (trip_df['chrg_cable_conn'] == 0) & (trip_df['speed'] == 0) & (trip_df['pack_current'] > 0)
        cond_regen = (trip_df['chrg_cable_conn'] == 0) & (trip_df['pack_current'] < 0)

        e_drive = calc_energy(trip_df['pack_volt'], trip_df['pack_current'], trip_df['delta_sec_power'])[cond_drive].sum()
        e_idle = calc_energy(trip_df['pack_volt'], trip_df['pack_current'], trip_df['delta_sec_power'])[cond_idle].sum()
        e_regen = calc_energy(trip_df['pack_volt'], -trip_df['pack_current'], trip_df['delta_sec_power'])[cond_regen].sum()

        e_trip_actual = e_drive + e_idle - e_regen
        e_trip_eff = (e_trip_actual / e_stored_diff) * 100 if e_stored_diff > 0 else np.nan

        trip_eff_list.append({
            'trip_id': tid,
            'e_trip_eff': e_trip_eff,
            'e_trip_actual': e_trip_actual,
            'e_stored_diff': e_stored_diff,
            'soc_0': soc_0,
            'soc_1': soc_1,
            'cell_volt_avg_0': start_volt,
            'cell_volt_avg_1': end_volt,
            'start_rest_time': trip_df.iloc[0]['time'],
            'end_rest_time': trip_df.iloc[-1]['time']
        })
    except Exception as e:
        print(f"[오류] Trip {tid} 처리 중 에러: {e}")
        continue

trip_eff_df = pd.DataFrame(trip_eff_list)
valid_trips = trip_eff_df.copy()


#trip_eff_df = pd.DataFrame(trip_eff_list, columns=['trip_id', 'e_trip_eff', 'e_trip_actual', 'e_stored_diff', 'soc_start', 'soc_end'])

# 이상 trip 제거
df['trip_id'] = df['trip_id'].where(df['trip_id'].isin(
    trip_eff_df[(trip_eff_df['e_trip_eff'] >= 85) & (trip_eff_df['e_trip_eff'] <= 115)]['trip_id']
))

# 전체 효율 계산
valid_eff_df = trip_eff_df[trip_eff_df['trip_id'].isin(df['trip_id'].dropna().unique())]
total_actual = valid_eff_df['e_trip_actual'].sum()
total_stored = valid_eff_df['e_stored_diff'].sum()

if total_stored > 0:
    total_eff = (total_actual / total_stored) * 100
else:
    total_eff = np.nan
    print("⚠️ Warning: total_stored가 0입니다. 전체 효율 계산 불가.")

print(f"\n✅ 전체 e_trip (유효 trip 기준) = {total_eff:.2f}%")


import matplotlib.pyplot as plt

df['power'] = calc_energy(df['pack_volt'], df['pack_current'], df['delta_sec_power'])

plt.figure(figsize=(15, 5))

# 전체 Power 흐름
plt.plot(df['time'], df['power'], label='Power (Wh)', color='gray', linewidth=0.5)

# Trip 구간 표시
trip_mask = df['trip_id'].notna()
plt.plot(df.loc[trip_mask, 'time'], df.loc[trip_mask, 'power'], 'r.', markersize=2, label='Trip 구간')

# Trip 경계선 표시
for t_id in df['trip_id'].dropna().unique():
    t_start_time = df.loc[df['trip_id'] == t_id, 'time'].iloc[0]
    plt.axvline(t_start_time, color='blue', linestyle='--', linewidth=0.5)

for idx, row in valid_trips.iterrows():
    start_rest_time = row['start_rest_time']
    end_rest_time = row['end_rest_time']
    soc_0 = row['soc_0']
    soc_1 = row['soc_1']
    volt_0 = row['cell_volt_avg_0']
    volt_1 = row['cell_volt_avg_1']

    power_start = df.loc[df['time'] == start_rest_time, 'power']
    power_end = df.loc[df['time'] == end_rest_time, 'power']

    if not power_start.empty:
        plt.plot(start_rest_time, power_start.values[0], 'bo', label='Start rest' if idx == 0 else "", markersize=6)
        plt.text(start_rest_time, power_start.values[0] + 100,
                 f"SOC₀={soc_0:.1%}, V={volt_0:.3f}V",
                 color='blue', fontsize=8, ha='center')

    if not power_end.empty:
        plt.plot(end_rest_time, power_end.values[0], 'g*', label='End rest' if idx == 0 else "", markersize=8)
        plt.text(end_rest_time, power_end.values[0] + 100,
                 f"SOC₁={soc_1:.1%}, V={volt_1:.3f}V",
                 color='green', fontsize=8, ha='center')


# 제목 및 축 설정
plt.title('Energy 흐름 + Trip 전후 rest 시점 전압 및 SOC 표시')
plt.xlabel('Time')
plt.ylabel('Energy (Wh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# trip 점 표시 (기존 plot → scatter로 변경)
plt.scatter(df.loc[trip_mask, 'time'], df.loc[trip_mask, 'power'],
            color='red', s=6, label='Trip 구간', alpha=0.6)

print(f"\n📊 trip_id 전체 개수: {df['trip_id'].nunique(dropna=True)}")
print(f"📊 trip_rest_volts에서 유효한 trip: {len(valid_trip_ids)}개")
print(f"📊 trip_eff_df 전체 길이: {len(trip_eff_df)}")
print(f"📊 최종 유효 trip (효율 85~115% 내): {len(valid_eff_df)}개")






for i, row in valid_trip_ids.iterrows():
    print(f"\n📍 Trip {i+1}")
    print(f"  ▶ Start time: {row['start_time']}")
    print(f"  ▶ End time: {row['end_time']}")
    print(f"  ▶ cell_volt_avg_0: {row['cell_volt_avg_0']:.4f} V")
    print(f"  ▶ SOC_0 (from OCV): {row['soc_0']:.4f}")
    print(f"  ▶ cell_volt_avg_1: {row['cell_volt_avg_1']:.4f} V")
    print(f"  ▶ SOC_1 (from OCV): {row['soc_1']:.4f}")
    print(f"  ▶ E_trip_actual: {row['E_trip_actual']:.2f} Wh")
    print(f"  ▶ E_stored_diff: {row['E_stored_0'] - row['E_stored_1']:.2f} Wh")
    print(f"  ▶ Efficiency e_trip: {row['e_trip'] * 100:.2f} %")


import matplotlib.pyplot as plt
plt.plot(ocv_curve, soc_curve, label='OCV-SOC Curve')
plt.scatter([row['cell_volt_avg_0'], row['cell_volt_avg_1']],
            [row['soc_0'], row['soc_1']],
            color='red', label='Trip Points')
plt.xlabel('OCV (V)')
plt.ylabel('SOC (%)')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 기존 그래프 그리는 부분 (power 시계열 예시)
plt.figure(figsize=(14, 6))
plt.plot(df['time'], df['power'], label='Power (전체)')

# trip 구간 빨간 점 (유효 trip만)
for idx, row in valid_trips.iterrows():
    trip_id = row['trip_id']

    # trip 데이터
    trip_df = df[df['trip_id'] == trip_id]

    # trip 구간 빨간 점 표시
    plt.scatter(trip_df['time'], trip_df['power'], color='red', s=6, alpha=0.5)

    # start rest 표시 (trip 시작 직전 rest 구간의 마지막 시점)
    start_rest_time = row['start_rest_time']
    power_start = df.loc[df['time'] == start_rest_time, 'power']
    if not power_start.empty:
        plt.plot(start_rest_time, power_start.values[0], 'bo', label='Start rest' if idx == 0 else "", markersize=6)
        plt.text(start_rest_time, power_start.values[0] + 100,
                 f"SOC₀={row['soc_0']:.2f}\nVolt={row['cell_volt_avg_0']:.3f}V",
                 color='blue', fontsize=8, ha='center')

    # end rest 표시 (trip 종료 2시간 후 지점)
    end_rest_time = row['end_rest_time']  # 이 시간은 trip 종료 + 2시간으로 미리 계산되어 있다고 가정
    power_end = df.loc[df['time'] == end_rest_time, 'power']
    if not power_end.empty:
        plt.plot(end_rest_time, power_end.values[0], 'g*', label='End rest' if idx == 0 else "", markersize=8)
        plt.text(end_rest_time, power_end.values[0] + 100,
                 f"SOC₁={row['soc_1']:.2f}\nVolt={row['cell_volt_avg_1']:.3f}V",
                 color='green', fontsize=8, ha='center')

plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.title('Trip 구간 및 Start/End Rest 지점 표시 (유효 trip만)')
plt.legend()
plt.grid(True)

# 시간 축 포맷 예쁘게 조정 (옵션)
plt.gcf().autofmt_xdate()
plt.show()
