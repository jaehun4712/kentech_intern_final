import pandas as pd
import numpy as np

# =============================
# 0. 데이터 불러오기 (병합된 df 사용)
# =============================
# df = pd.read_csv("merged_11_12.csv", parse_dates=['timestamp'])  # 예시

# =============================
# 1. timestamp 기준 정렬 및 delta_sec 계산
# =============================
df = df.sort_values('timestamp').reset_index(drop=True)
df['delta_sec'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

# =============================
# 2. rest 구간 판단
# =============================
rest_speed_zero = df['speed'] == 0
rest_cable_unplugged = df['chrg_cable_conn'] == 0
rest_low_current = df['pack_current'].abs() < 1
df['is_rest'] = rest_speed_zero & rest_cable_unplugged & rest_low_current

# =============================
# 3. trip 시작/종료 조건 및 trip_id 부여
# =============================
df['prev_is_rest'] = df['is_rest'].shift(1, fill_value=True)
df['next_is_rest'] = df['is_rest'].shift(-1, fill_value=True)
df['trip_start'] = (~df['is_rest']) & (df['prev_is_rest'])
df['trip_end'] = (~df['is_rest']) & (df['next_is_rest'])

trip_counter = 0
in_trip = False
df['trip_id'] = np.nan
for idx, row in df.iterrows():
    if row['trip_start']:
        in_trip = True
        trip_counter += 1
    if in_trip:
        df.at[idx, 'trip_id'] = trip_counter
    if row['trip_end']:
        in_trip = False

df['trip_id'] = df['trip_id'].ffill()

# =============================
# 4. 유효 trip 필터링 (앞뒤 rest > 7200초)
# =============================
trip_valid = []
for trip_id, group in df.groupby('trip_id'):
    if pd.isna(trip_id):
        continue
    start_idx = group.index.min() - 1
    end_idx = group.index.max() + 1
    rest_before = (start_idx >= 0) and df.loc[start_idx, 'is_rest'] and df.loc[start_idx, 'delta_sec'] > 7200
    rest_after = (end_idx < len(df)) and df.loc[end_idx, 'is_rest'] and df.loc[end_idx, 'delta_sec'] > 7200
    if rest_before and rest_after:
        trip_valid.append(trip_id)

df['valid_trip'] = df['trip_id'].isin(trip_valid)

# =============================
# 5. 저장 에너지 추정 함수 정의 (OCV-SOC 곡선 기반 또는 근사)
# =============================
def estimate_E_stored(soc):
    Q = 56.47  # 셀 기준 용량 (Ah)
    V_avg = 3.7 + 0.5 * (1 - soc)  # 예시용 평균 전압 함수
    return Q * V_avg * soc  # Wh

# =============================
# 6. trip별 e_trip 계산
# =============================
trip_results = []
for trip_id in trip_valid:
    trip_df = df[df['trip_id'] == trip_id].copy()

    soc_start = trip_df['soc'].iloc[0]
    soc_end = trip_df['soc'].iloc[-1]
    E_stored_diff = estimate_E_stored(soc_start) - estimate_E_stored(soc_end)

    trip_df['power'] = trip_df['pack_voltage'] * trip_df['pack_current']
    trip_df['delta_sec_power'] = trip_df['delta_sec'].where(trip_df['delta_sec'] < 300, 0)
    E_actual = (trip_df['power'] * trip_df['delta_sec_power']).sum() / 3600

    e_trip_eff = E_actual / E_stored_diff if E_stored_diff > 0 else np.nan

    trip_results.append({
        'trip_id': trip_id,
        'e_trip_eff': e_trip_eff * 100,
        'e_trip_actual': E_actual,
        'e_stored_diff': E_stored_diff,
    })

trip_df_result = pd.DataFrame(trip_results)

# =============================
# 7. 이상 trip 제거 및 최종 결과
# =============================
trip_df_result['abnormal'] = (trip_df_result['e_trip_eff'] < 60) | (trip_df_result['e_trip_eff'] > 110)
normal_trips = trip_df_result[~trip_df_result['abnormal']]

mean_eff = normal_trips['e_trip_eff'].mean()
total_actual = normal_trips['e_trip_actual'].sum()
total_stored = normal_trips['e_stored_diff'].sum()

print(f"\n✅ 평균 e_trip 효율 (정상 trip 기준): {mean_eff:.2f}%")
print(f"🔋 총 실제 사용 에너지: {total_actual:.1f} Wh")
print(f"🔋 총 저장 기반 추정 에너지: {total_stored:.1f} Wh")

print("\n🚨 이상 trip 목록:")
print(trip_df_result[trip_df_result['abnormal']][['trip_id', 'e_trip_eff', 'e_trip_actual', 'e_stored_diff']])
