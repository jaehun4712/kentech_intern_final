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
bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\3week_data\bms_01241225206_2023-08.csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

# ----------------------------
# BMS 데이터 로드 및 시간 처리
# ----------------------------
bms_df = pd.read_csv(bms_file)
bms_df['cell_volt_list']=bms_df['cell_volt_list'].astype(str)
bms_df['cell_list']=bms_df['cell_volt_list'].apply(lambda x:[float(i) for i in x.split(',') if i.strip()])

bms_df['cell_count']=bms_df['cell_list'].apply(len)
bms_df['avg_cell_voltage']=bms_df['cell_list'].apply(lambda x: sum(x)/len(x) if len(x)>0 else None)

#Qmax_per_cell=56.47
#bms_df['Qmax_total']=bms_df['cell_count']*Qmax_per_cell




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
        trip_eff_list.append((tid, np.nan, e_trip_actual, e_stored_diff))
        print(f"[Trip {tid}] SOC 변화 적음 → e_trip 계산 생략")

# ----------------------------
# ----------------------------
# 비정상 Trip 필터링
# 기준: e_trip_eff가 120% 이상 또는 70% 이하
# ----------------------------
trip_eff_df = pd.DataFrame(trip_eff_list, columns=['trip_id', 'e_trip_eff', 'e_trip_actual', 'e_stored_diff'])

abnormal_trips = trip_eff_df[
    (trip_eff_df['e_trip_eff'] > 120) | (trip_eff_df['e_trip_eff'] < 70)
]

print("🚨 이상(비정상) Trip 목록:")
print(abnormal_trips[['trip_id', 'e_trip_eff', 'e_trip_actual', 'e_stored_diff']].to_string(index=False))

# 필요 시 이상 Trip ID만 추출
abnormal_trip_ids = abnormal_trips['trip_id'].tolist()



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

# ----------------------------
# 전체 평균 효율
# ----------------------------
total_actual = trip_eff_df['e_trip_actual'].sum()
total_stored = trip_eff_df['e_stored_diff'].sum()
total_eff = (total_actual / total_stored) * 100
print(f"\n✅ 전체 e_trip = {total_eff:.2f}%")
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

plt.title('Power 흐름 및 Trip 경계')
plt.xlabel('Time')
plt.ylabel('Power (Wh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





# 이상 trip 구간의 BMS 데이터만 추출
abnormal_bms_segments = pd.DataFrame()

for tid in abnormal_trip_ids:
    segment = df[df['trip_id'] == tid].copy()
    abnormal_bms_segments = pd.concat([abnormal_bms_segments, segment], axis=0)

# 정렬 및 인덱스 초기화
abnormal_bms_segments = abnormal_bms_segments.sort_values('time').reset_index(drop=True)

# 원하는 컬럼만 필터링 (예시: 시간, SOC, 전류, 전압)
cols_to_check = ['time', 'trip_id', 'soc', 'pack_current', 'pack_volt', 'delta_sec', 'delta_sec_power']
abnormal_bms_segments = abnormal_bms_segments[cols_to_check]

# 결과 확인
print(abnormal_bms_segments.head())

abnormal_bms_segments.to_excel("이상_trip_BMS_데이터.xlsx", index=False)
print("✅ 이상 trip BMS 데이터가 '이상_trip_BMS_데이터.xlsx'로 저장되었습니다.")

import matplotlib.pyplot as plt

# 이상 trip별 시각화
for tid in abnormal_trip_ids:
    trip_df = df[df['trip_id'] == tid].copy()

    if trip_df.empty:
        continue

    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"📉 이상 Trip ID: {tid} 시각화", fontsize=16)

    # 1. SOC
    axs[0].plot(trip_df['time'], trip_df['soc'], label='SOC (%)', color='blue')
    axs[0].set_ylabel('SOC (%)')
    axs[0].grid(True)
    axs[0].legend(loc='upper right')

    # 2. 팩 전류
    axs[1].plot(trip_df['time'], trip_df['pack_current'], label='Pack Current (A)', color='red')
    axs[1].set_ylabel('Current (A)')
    axs[1].grid(True)
    axs[1].legend(loc='upper right')

    # 3. 전력 (Wh)
    axs[2].plot(trip_df['time'], trip_df['power'], label='Power (Wh)', color='green')
    axs[2].set_ylabel('Power (Wh)')
    axs[2].set_xlabel('Time')
    axs[2].grid(True)
    axs[2].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
