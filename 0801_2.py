import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

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
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)
bms_df['delta_sec'] = bms_df['delta_sec_raw']
bms_df['date'] = bms_df['time'].dt.date
bms_df['delta_sec_power'] = bms_df['delta_sec_raw'].where(
    (bms_df['delta_sec_raw'] <= 300) & (bms_df['date'] == bms_df['date'].shift(1)), 0
)

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
inv_ocv_func = interp1d(ocv_vals, soc_vals, kind='linear', fill_value="extrapolate")

Qmax = 56.47
n_cells = 192
parallel = 2

# ----------------------------
# 적분 누적 테이블 캐싱하여 빠른 E_stored 계산 함수
# ----------------------------
soc_grid = np.linspace(0, 1, 1001)
ocv_vals_grid = ocv_func(soc_grid)
cumulative_integral = cumtrapz(ocv_vals_grid, soc_grid, initial=0)
estored_table = Qmax * n_cells * parallel * cumulative_integral  # [J]

def compute_estored_fast(soc):
    soc = np.clip(soc, 0, 1)
    return np.interp(soc, soc_grid, estored_table)

# ----------------------------
# Trip ID 부여
# ----------------------------
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
        if i > 0 and df.loc[i, 'speed'] == 0 and df.loc[i - 1, 'speed'] > 0:
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

# ----------------------------
# 조건 만족 Trip 필터링 (rest 전후 모두 2시간 이상)
# ----------------------------
valid_trip_info = []

for tid in df['trip_id'].dropna().unique():
    tid = int(tid)
    trip_df = df[df['trip_id'] == tid].copy()
    if trip_df.empty:
        continue

    trip_start_idx = trip_df.index[0]
    trip_end_idx = trip_df.index[-1]

    # trip 전 rest 확인
    pre_rest_idx_candidates = df.index[(df.index < trip_start_idx) & (df['speed'] == 0) & (df['chrg_cable_conn'] == 0)]
    if len(pre_rest_idx_candidates) == 0:
        continue
    pre_rest_idx = pre_rest_idx_candidates[-1]
    pre_rest_start_idx = pre_rest_idx
    while pre_rest_start_idx > 0 and df.loc[pre_rest_start_idx, 'speed'] == 0 and df.loc[pre_rest_start_idx, 'chrg_cable_conn'] == 0:
        pre_rest_start_idx -= 1
    pre_rest_duration = df.loc[pre_rest_start_idx + 1:pre_rest_idx, 'delta_sec'].sum()
    if pre_rest_duration < 7200:
        continue

    # trip 후 rest 확인
    post_rest_idx_candidates = df.index[(df.index > trip_end_idx) & (df['speed'] == 0) & (df['chrg_cable_conn'] == 0)]
    if len(post_rest_idx_candidates) == 0:
        continue
    post_rest_idx = post_rest_idx_candidates[0]
    post_rest_start_idx = post_rest_idx
    while post_rest_start_idx > 0 and df.loc[post_rest_start_idx, 'speed'] == 0 and df.loc[post_rest_start_idx, 'chrg_cable_conn'] == 0:
        post_rest_start_idx -= 1
    post_rest_duration = df.loc[post_rest_start_idx + 1:post_rest_idx, 'delta_sec'].sum()
    if post_rest_duration < 7200:
        continue

    # OCV SOC 계산
    rest_voltage = df.loc[post_rest_idx, 'pack_volt']
    rest_voltage_per_cell = rest_voltage / n_cells
    ocv_soc_estimate = inv_ocv_func(rest_voltage_per_cell) * 100

    # 전류=0 조건 확인 (rest 시점 전후 ±5분 내)
    window_df = df.loc[max(post_rest_idx - 10, 0): min(post_rest_idx + 10, len(df)-1)]
    is_idle_zero_current = window_df['pack_current'].abs().max() < 1.0  # 1A 미만이면 0으로 간주

    # E_stored 계산 (trip 종료 시점과 rest 이후 시점)
    soc_trip_end = df.loc[trip_end_idx, 'soc'] / 100
    soc_rest = ocv_soc_estimate / 100
    estored_trip_end = compute_estored_fast(soc_trip_end)
    estored_rest = compute_estored_fast(soc_rest)
    estored_gap = estored_trip_end - estored_rest

    valid_trip_info.append({
        'trip_id': tid,
        'trip_start_idx': trip_start_idx,
        'trip_end_idx': trip_end_idx,
        'trip_start_time': df.loc[trip_start_idx, 'time'],
        'trip_end_time': df.loc[trip_end_idx, 'time'],
        'rest_after_idx': post_rest_idx,
        'rest_after_time': df.loc[post_rest_idx, 'time'],
        'estored_gap': estored_gap,
        'is_idle_zero_current': is_idle_zero_current,
    })

valid_trip_df = pd.DataFrame(valid_trip_info)

# ----------------------------
# 효율 계산
# ----------------------------
df['power'] = df['pack_volt'] * df['pack_current']
df['energy_Wh'] = df['power'] * df['delta_sec_power'] / 3600  # Wh 단위 변환

trip_efficiency_list = []

for idx, row in valid_trip_df.iterrows():
    tid = row['trip_id']
    trip_segment = df[df['trip_id'] == tid]

    E_actual_Wh = trip_segment['energy_Wh'].sum()
    E_stored_diff_Wh = abs(row['estored_gap']) / 3600  # J → Wh 환산

    if E_actual_Wh == 0:
        eff = np.nan
    else:
        eff = E_stored_diff_Wh / E_actual_Wh

    trip_efficiency_list.append({'trip_id': tid, 'efficiency': eff})

eff_df = pd.DataFrame(trip_efficiency_list)

print(f"trip_id 개수: {df['trip_id'].nunique()}")
print(f"유효 trip 개수: {len(valid_trip_df)}")
print(valid_trip_df[['trip_id', 'trip_start_time', 'rest_after_time']])

# ----------------------------
# 시각화 (에너지 vs 시간, trip 시작과 2시간 이후 rest 구간 표시)
# ----------------------------
plt.figure(figsize=(14, 7))

# y축: 에너지, x축: 시간
plt.plot(df['time'], compute_estored_fast(df['soc'] / 100), label='Stored Energy (J)', color='black', alpha=0.7)

# trip 구간 빨간 점 표시
for idx, trip in valid_trip_df.iterrows():
    plt.scatter(df.loc[trip['trip_start_idx']:trip['trip_end_idx'], 'time'],
                compute_estored_fast(df.loc[trip['trip_start_idx']:trip['trip_end_idx'], 'soc'] / 100),
                color='red', s=10, label='Valid Trip' if idx == 0 else "")

    # trip 시작 후 2시간 뒤 rest 지점 파란색 점 표시
    plt.scatter(trip['rest_after_time'], compute_estored_fast(inv_ocv_func(df.loc[trip['rest_after_idx'], 'pack_volt'] / n_cells)),
                color='blue', s=20, label='Rest after 2h' if idx == 0 else "")

plt.xlabel('Time')
plt.ylabel('Stored Energy (J)')
plt.legend()
plt.title('Trip and Rest Energy Analysis with Fast Computation')
plt.grid(True)
plt.show()
