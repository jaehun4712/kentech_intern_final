####최종 e_trip계산 파일 ####



import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 파일 경로
# ----------------------------
bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\EV6_1year\bms_01241225206_2023-08 (1).csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

# ----------------------------
# BMS 데이터 로드 및 시간 처리
# ----------------------------
bms_df = pd.read_csv(bms_file)
bms_df['time'] = pd.to_datetime(bms_df['time'], format='mixed', errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)



# 날짜 및 delta_sec 계산

bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)
bms_df['delta_sec'] = bms_df['delta_sec_raw']  # trip 구분용


bms_df['date'] = bms_df['time'].dt.date

# ⛽ 전력 계산 전용 delta_sec (300초 초과 or 날짜 변경 시 0)
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

Qmax = 56.47
n_cells = 192
parallel = 2

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
trip_id = 0
in_trip = False
trip_ids = [np.nan] * len(df)

i = 0
while i < len(df):
    if not in_trip:
        if df.loc[i, 'speed'] > 0 and df.loc[i, 'chrg_cable_conn'] == 0:
            trip_start = i
            in_trip = True
    else:
        # 주행 → 정지 전환 확인
        if df.loc[i, 'speed'] == 0 and df.loc[i - 1, 'speed'] > 0:
            stop_idx = i
            j = i + 1
            while j < len(df) and df.loc[j, 'speed'] == 0:
                if df.loc[j, 'delta_sec'] > 300:
                    df.loc[trip_start:stop_idx - 1, 'trip_id'] = trip_id
                    trip_id += 1
                    in_trip = False
                    i = stop_idx - 1  # 다시 탐색 시작을 정확히 조정
                    break
                j += 1
        # 충전 시작하면 trip 종료
        if df.loc[i, 'chrg_cable_conn'] == 1:
            df.loc[trip_start:i, 'trip_id'] = trip_id
            trip_id += 1
            in_trip = False
    i += 1

valid_trip_ids = []
trip_ids = df['trip_id'].dropna().unique()

for tid in trip_ids:
    trip_df = df[df['trip_id'] == tid]
    trip_start_idx = trip_df.index[0]
    trip_end_idx = trip_df.index[-1]

    # ▶ trip 시작 전 rest 조건 확인
    if trip_start_idx == 0:
        continue
    prev_idx = trip_start_idx - 1
    while prev_idx >= 0 and df.loc[prev_idx, 'speed'] == 0 and df.loc[prev_idx, 'chrg_cable_conn'] == 0:
        if df.loc[prev_idx + 1, 'delta_sec'] > 7200 or df.loc[prev_idx + 1, 'date'] != df.loc[prev_idx, 'date']:
            break
        prev_idx -= 1
    else:
        continue  # 시작 전 rest 조건 불충족

    # ▶ trip 종료 후 rest 조건 확인
    if trip_end_idx >= len(df) - 1:
        continue
    next_idx = trip_end_idx + 1
    while next_idx < len(df) and df.loc[next_idx, 'speed'] == 0 and df.loc[next_idx, 'chrg_cable_conn'] == 0:
        if df.loc[next_idx, 'delta_sec'] > 7200 or df.loc[next_idx, 'date'] != df.loc[next_idx - 1, 'date']:
            break
        next_idx += 1
    else:
        continue  # 종료 후 rest 조건 불충족

    valid_trip_ids.append(tid)

# 유효하지 않은 trip 제거
df['trip_id'] = df['trip_id'].where(df['trip_id'].isin(valid_trip_ids))

# ----------------------------
# Trip 단위 e_trip 계산
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

    e_drive = calc_energy(trip_df['pack_volt'], trip_df['pack_current'], trip_df['delta_sec_power'])[cond_drive_discharge].sum()
    e_idle = calc_energy(trip_df['pack_volt'], trip_df['pack_current'], trip_df['delta_sec_power'])[cond_idle_discharge].sum()
    e_regen = calc_energy(trip_df['pack_volt'], -trip_df['pack_current'], trip_df['delta_sec_power'])[cond_regen_charge].sum()

    e_trip_actual = e_drive + e_idle - e_regen

    if e_stored_diff > 0:
        e_trip_eff = (e_trip_actual / e_stored_diff) * 100
        trip_eff_list.append((tid, e_trip_eff, e_trip_actual, e_stored_diff))
        print(f"[Trip {tid}] e_trip = {e_trip_eff:.2f}% ({e_trip_actual:.1f} / {e_stored_diff:.1f} Wh)")
    else:
       # trip_eff_list.append((tid, np.nan, e_trip_actual, e_stored_diff))
        print(f"[Trip {tid}] SOC 변화 적음 → e_trip 계산 생략")

# ----------------------------
# ----------------------------
# 비정상 Trip 필터링
# 기준: e_trip_eff가 120% 이상 또는 70% 이하
# ----------------------------
trip_eff_df = pd.DataFrame(trip_eff_list, columns=['trip_id', 'e_trip_eff', 'e_trip_actual', 'e_stored_diff'])

abnormal_trips = trip_eff_df[
    (trip_eff_df['e_trip_eff'] > 115) | (trip_eff_df['e_trip_eff'] < 85)
]


print("🚨 이상(비정상) Trip 목록:")
print(abnormal_trips[['trip_id', 'e_trip_eff', 'e_trip_actual', 'e_stored_diff']].to_string(index=False))

# 필요 시 이상 Trip ID만 추출
abnormal_trip_ids = abnormal_trips['trip_id'].tolist()

# ⚠️ 비정상 trip도 최종 trip_id에서 제외
valid_normal_trip_ids = trip_eff_df[
    (trip_eff_df['trip_id'].isin(valid_trip_ids)) &
    (~trip_eff_df['trip_id'].isin(abnormal_trip_ids))
]['trip_id'].tolist()

# trip_id 갱신 (정상 + rest 조건 만족한 trip만 유지)
df['trip_id'] = df['trip_id'].where(df['trip_id'].isin(valid_normal_trip_ids))




# ----------------------------
# 효율 시각화
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(trip_eff_df['trip_id'], trip_eff_df['e_trip_eff'], marker='o', linestyle='-', color='blue')
plt.title('Trip 별 e_trip 효율')
plt.xlabel('Trip ID')
plt.ylabel('Efficiency (%)')
plt.grid(True)
plt.tight_layout()
plt.show()



# ✅ 유효한 trip만 필터링 (rest 조건 + 정상 효율 범위)
filtered_trip_eff_df = trip_eff_df[trip_eff_df['trip_id'].isin(valid_normal_trip_ids)]

# ⬇ 이걸로 전체 효율 계산
total_actual = filtered_trip_eff_df['e_trip_actual'].sum()
total_stored = filtered_trip_eff_df['e_stored_diff'].sum()
total_eff = (total_actual / total_stored) * 100

print(f"\n✅ 전체 e_trip (유효 trip 기준) = {total_eff:.2f}%")
print(f"    ⤷ total actual: {total_actual:,.1f} Wh")
print(f"    ⤷ total stored: {total_stored:,.1f} Wh")



# ----------------------------
# Power 흐름 시각화 + trip 경계 표시
# ----------------------------
df['power'] = calc_energy(df['pack_volt'], df['pack_current'], df['delta_sec_power'])

plt.figure(figsize=(15, 5))
plt.plot(df['time'], df['power'], label='Power (Wh)', color='gray', linewidth=0.5)

trip_mask = df['trip_id'].notna()
plt.plot(df.loc[trip_mask, 'time'], df.loc[trip_mask, 'power'], 'r.', markersize=2, label='Trip 구간')

for t_id in df['trip_id'].dropna().unique():
    t_start_time = df.loc[df['trip_id'] == t_id, 'time'].iloc[0]
    plt.axvline(t_start_time, color='blue', linestyle='--', linewidth=0.5)

plt.title('Energy 흐름 및 Trip 경계')
plt.xlabel('Time')
plt.ylabel('Energy (Wh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



