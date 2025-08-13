import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 파일 경로
# ----------------------------
bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\EV6_1year\bms_merged_sorted.csv"
ocv_file = r"C:\\Users\\OWNER\\Desktop\\EV6_2301\\NE_Cell_Characterization_performance.xlsx"

# ----------------------------
# BMS 데이터 로드 및 시간 처리
# ----------------------------
bms_df = pd.read_csv(bms_file)
bms_df['time'] = pd.to_datetime(bms_df['time'], format='mixed', errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)

# 날짜 및 delta_sec 계산
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
trip_ids = df['trip_id'].dropna().unique()

for tid in trip_ids:
    trip_df = df[df['trip_id'] == tid]
    trip_start_idx = trip_df.index[0]
    trip_end_idx = trip_df.index[-1]

    if trip_start_idx == 0:
        continue
    prev_idx = trip_start_idx - 1
    while prev_idx >= 0 and df.loc[prev_idx, 'speed'] == 0 and df.loc[prev_idx, 'chrg_cable_conn'] == 0:
        if df.loc[prev_idx + 1, 'delta_sec'] > 7200 or df.loc[prev_idx + 1, 'date'] != df.loc[prev_idx, 'date']:
            break
        prev_idx -= 1
    else:
        continue

    if trip_end_idx >= len(df) - 1:
        continue
    next_idx = trip_end_idx + 1
    while next_idx < len(df) and df.loc[next_idx, 'speed'] == 0 and df.loc[next_idx, 'chrg_cable_conn'] == 0:
        if df.loc[next_idx, 'delta_sec'] > 7200 or df.loc[next_idx, 'date'] != df.loc[next_idx - 1, 'date']:
            break
        next_idx += 1
    else:
        continue

    valid_trip_ids.append(tid)

df['trip_id'] = df['trip_id'].where(df['trip_id'].isin(valid_trip_ids))

# 급가속/감속 이벤트 정의
a_pos_threshold = 1.5
a_neg_threshold = -1.5
df['event'] = 'normal'
df.loc[df['acceleration'] >= a_pos_threshold, 'event'] = 'hard_accel'
df.loc[df['acceleration'] <= a_neg_threshold, 'event'] = 'hard_brake'

# trip별 e_trip 계산

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
    cond_regen = (trip_df['chrg_cable_conn'] == 0) & (trip_df['pack_current'] < 0)

    e_drive = calc_energy(trip_df['pack_volt'], trip_df['pack_current'], trip_df['delta_sec_power'])[cond_drive_discharge].sum()
    e_idle = calc_energy(trip_df['pack_volt'], trip_df['pack_current'], trip_df['delta_sec_power'])[cond_idle_discharge].sum()
    e_regen = calc_energy(trip_df['pack_volt'], -trip_df['pack_current'], trip_df['delta_sec_power'])[cond_regen].sum()

    e_trip_actual = e_drive + e_idle - e_regen

    if e_stored_diff > 0:
        e_trip_eff = (e_trip_actual / e_stored_diff) * 100
        total_idle_time = trip_df.loc[trip_df['speed'] == 0, 'delta_sec'].sum()
        trip_eff_list.append((tid, e_trip_eff, e_trip_actual, e_stored_diff, total_idle_time))
    else:
        continue

trip_eff_df = pd.DataFrame(trip_eff_list, columns=['trip_id', 'e_trip_eff', 'e_trip_actual', 'e_stored_diff', 'total_idle_time'])

# 비정상 trip 제거
total_eff = trip_eff_df[(trip_eff_df['e_trip_eff'] <= 115) & (trip_eff_df['e_trip_eff'] >= 85)]

# 이벤트 집계 및 병합
event_counts = df[df['trip_id'].isin(total_eff['trip_id'])].groupby(['trip_id', 'event']).size().unstack(fill_value=0).reset_index()
event_counts['total_events'] = event_counts.get('hard_accel', 0) + event_counts.get('hard_brake', 0)

result_df = pd.merge(total_eff, event_counts, on='trip_id', how='left')

# ----------------------------
# 하위 20% trip만 필터링하여 별도 분석
# ----------------------------
# 1. e_trip_eff 기준 하위 20% 추출
threshold_20 = result_df['e_trip_eff'].quantile(0.2)
low_eff_df = result_df[result_df['e_trip_eff'] <= threshold_20]

print(f"\n🔍 e_trip_eff 하위 20% 기준값: {threshold_20:.2f}")
print(f"하위 20% trip 수: {len(low_eff_df)}개")

# 2. 하위 20% trip에서 상관관계 분석
print("\n📉 [하위 20% trip] 상관계수 (Correlation Coefficients):")
print(low_eff_df[['e_trip_eff', 'hard_accel', 'hard_brake', 'total_events', 'total_idle_time']].corr()['e_trip_eff'])

# 3. 시각화 - 이벤트 vs e_trip_eff (하위 20%)
sns.lmplot(data=low_eff_df, x='total_events', y='e_trip_eff')
plt.title('[하위 20%] 이벤트 횟수 vs e_trip 효율')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. 시각화 - 정지시간 vs e_trip_eff (하위 20%)
sns.lmplot(data=low_eff_df, x='total_idle_time', y='e_trip_eff')
plt.title('[하위 20%] 정지 시간 vs e_trip 효율')
plt.grid(True)
plt.tight_layout()
plt.show()

# 급가속/감속 시각화 (시간 흐름상)
# ...
# 상관계수 출력
print("\n📈 상관계수 (Correlation Coefficients):")
print(result_df[['e_trip_eff', 'hard_accel', 'hard_brake', 'total_events', 'total_idle_time']].corr()['e_trip_eff'])


# 급가속/감속 시각화 (시간 흐름상)
plt.figure(figsize=(15, 5))
plt.plot(df['time'], df['speed'], label='Speed', alpha=0.5)
plt.scatter(df[df['event'] == 'hard_accel']['time'], df[df['event'] == 'hard_accel']['speed'], color='red', label='Hard Accel', s=10)
plt.scatter(df[df['event'] == 'hard_brake']['time'], df[df['event'] == 'hard_brake']['speed'], color='blue', label='Hard Brake', s=10)
plt.legend()
plt.title('시간 흐름상 급가속/급감속 이벤트')
plt.xlabel('시간')
plt.ylabel('속도')
plt.grid(True)
plt.tight_layout()
plt.show()

# 효율 vs 이벤트 횟수
sns.lmplot(data=result_df, x='total_events', y='e_trip_eff')
plt.title('이벤트 횟수 vs e_trip 효율')
plt.grid(True)
plt.tight_layout()
plt.show()

# 효율 vs 정지 시간
sns.lmplot(data=result_df, x='total_idle_time', y='e_trip_eff')
plt.title('정지 시간 vs e_trip 효율')
plt.grid(True)
plt.tight_layout()
plt.show()

# 상관계수 출력
print("\n📈 상관계수 (Correlation Coefficients):")
print(result_df[['e_trip_eff', 'hard_accel', 'hard_brake', 'total_events', 'total_idle_time']].corr()['e_trip_eff'])


