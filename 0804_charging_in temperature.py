import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 파일 경로
# ----------------------------
bms_file = r"C:\Users\OWNER\Desktop\EV6_2301\EV6_1year\bms_merged_sorted.csv"
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

# ----------------------------
# BMS 데이터 로드 및 시간 처리
# ----------------------------
bms_df = pd.read_csv(bms_file)
bms_df['time'] = pd.to_datetime(bms_df['time'], format='mixed', errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)

# delta_sec 계산 및 segment 구분
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)
gap_indices = bms_df.index[bms_df['delta_sec_raw'] > 60].tolist()#시간간격이 60초보다 크면 bms꺼짐 or 날짜변경으로
                                                                   #판단하여 단절지점 분류

segment_id = 0
segments = []
for i in range(len(bms_df)):
    segments.append(segment_id)
    if i in gap_indices:
        segment_id += 1
bms_df['segment_id'] = segments

# 각 세그먼트에서 다시 delta_sec 계산
bms_df['delta_sec'] = (
    bms_df.groupby('segment_id')['time']#그룹화한 segment_id내에서 시간간격 계산
    .diff()
    .dt.total_seconds() #시간간격을 초단위로 변경
    .fillna(0)
)

# delta_sec > 10 제거용 마스크
bms_df['valid'] = bms_df['delta_sec'] <= 10

# ----------------------------
# OCV 곡선 처리
# ----------------------------
ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
start_row = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index[0] + 1
soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]] #C_rate=0.05C로 수정
soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float)

soc_vals = soc_ocv_data['SOC'].values/100
ocv_vals = soc_ocv_data['OCV'].values

valid_mask = (~np.isnan(soc_vals)) & (~np.isnan(ocv_vals))
soc_vals = soc_vals[valid_mask]
ocv_vals = ocv_vals[valid_mask]

sort_idx = np.argsort(soc_vals)
soc_vals = soc_vals[sort_idx]
ocv_vals = ocv_vals[sort_idx]

ocv_func = interp1d(soc_vals, ocv_vals, kind='linear', fill_value="extrapolate")

# ----------------------------
# SOC 기반 에너지 계산
# ----------------------------
Qmax = 56.47

bms_columns = [col.lower() for col in bms_df.columns]
soc_col = next((col for col in bms_columns if 'soc' in col), None)

if soc_col:
    soc_series = bms_df.loc[bms_df['valid'], bms_df.columns[bms_columns.index(soc_col)]].dropna()
    soc_0 = soc_series.iloc[0]/100
    soc_1 = soc_series.iloc[-1]/100

    def compute_estored(soc):
        if soc <= 0.001:
            return 0
        try:
            result, _ = quad(ocv_func, 0, soc, limit=200)
            return Qmax * result*192*2
        except Exception as e:
            print(f"[적분 실패] SOC: {soc:.4f} → 오류: {e}")
            return 0

    e_stored_0 = compute_estored(soc_0)
    e_stored_1 = compute_estored(soc_1)
    e_stored_diff = e_stored_0 - e_stored_1

    print(f"SOC_0: {soc_0 * 100:.1f}%, E_stored_0: {e_stored_0:.2f} Wh")
    print(f"SOC_1: {soc_1 * 100:.1f}%, E_stored_1: {e_stored_1:.2f} Wh")
    print(f"E_stored_0 - E_stored_1: {e_stored_diff:.2f} Wh")
else:
    print("⚠️ SOC 컬럼을 찾을 수 없습니다.")
    exit()

# ----------------------------
# 조건별 에너지 계산
# ----------------------------
df = bms_df.copy()
df.columns = [col.lower() for col in df.columns]

cable_col = 'chrg_cable_conn'
speed_col = 'speed'
current_col = 'pack_current'
voltage_col = 'pack_volt'
time_interval = 'delta_sec'

def calc_energy(voltage, current, time_s):
    return (voltage * current * time_s) / 3600

# 조건 정의 (모두 valid 구간만 포함)
cond_discharge_drive = df['valid'] & (df[cable_col] == 0) & (df[speed_col] > 0) & (df[current_col] > 0)
cond_discharge_idle  = df['valid'] & (df[cable_col] == 0) & (df[speed_col] == 0) & (df[current_col] > 0)
cond_trip_charge     = df['valid'] & (df[cable_col] == 0) & (df[current_col] < 0) #회생제동 조건 수정
cond_charging        = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] < 0)
cond_charging_idle   = df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] > 0)

# 에너지 계산
E_trip_discharge_drive = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_discharge_drive].sum()
E_trip_discharge_idle  = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_discharge_idle].sum()
E_trip_discharge = E_trip_discharge_drive + E_trip_discharge_idle

E_trip_charge = calc_energy(df[voltage_col], -df[current_col], df[time_interval])[cond_trip_charge].sum()
E_real_charging = calc_energy(df[voltage_col], -df[current_col], df[time_interval])[cond_charging].sum()
E_charging_idle = calc_energy(df[voltage_col], df[current_col], df[time_interval])[cond_charging_idle].sum()
E_charging = E_real_charging - E_charging_idle
E_trip_net = E_trip_discharge - E_trip_charge

# ----------------------------
# 출력
# ----------------------------
print(f".................................")
print(f"E_trip_discharge_drive: {E_trip_discharge_drive:.2f} Wh")
print(f"E_trip_discharge_idle: {E_trip_discharge_idle:.2f} Wh")
print(f"E_trip_discharge = E_trip_discharge_drive + E_trip_discharge_idle: {E_trip_discharge:.2f} Wh")
print(f"E_trip_charge (regen): {E_trip_charge:.2f} Wh")
print(f"E_trip_net = E_trip_discharge - E_trip_charge (regen) : {E_trip_net:.2f} Wh")
print(f".................................")
print(f"E_real_charging : {E_real_charging:.2f} Wh")
print(f"E_charging_idle : {E_charging_idle:.2f} Wh")
print(f"E_charging = E_real_charging - E_charging_idle: {E_charging:.2f} Wh")
print(f".................................")

if (E_charging + e_stored_diff) > 0:
    efficiency1 = E_trip_net / (E_charging + e_stored_diff) * 100
    print(f"Efficiency e1 = {efficiency1:.4f}%")

    efficiency2 = (E_trip_net + e_stored_1) / (E_charging + e_stored_0) * 100
    print(f"Efficiency e2 = {efficiency2:.4f}%")
    print(f".................................")
else:
    print("⚠️ Efficiency 계산 불가: 분모가 0 또는 음수")

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad



# 1. 충전 조건: 충전 케이블 연결 & 전류 < 0
charging_mask =   df['valid'] & (df[cable_col] == 1) & (df[speed_col] == 0) & (df[current_col] < 0)
#charging_mask=충전조건

# 2. 충전 구간 구분: 연속된 True 값으로 구간 번호 생성
df['chg_segment'] = (charging_mask != charging_mask.shift()).cumsum()
#한칸씩 데이터를 밀어서 이전값과 비교하여 값이 바뀌는 지점이 충전을 만족하는 구간과 아닌 구간의 경계
#바뀌는 지점을 카운팅 하고 구간 번호 생성
df.loc[~charging_mask, 'chg_segment'] = np.nan
#충전조건이 아닌지점, False인 지점은 NaN.즉 충전구간에는 같은 번호가 부여되고, 아닌 지점은 NaN

# 3. 충전 구간 그룹화
chg_groups = df[df['chg_segment'].notna()].groupby('chg_segment')
#chg_segment가 NaN이 아닌 것들을 chg_groups로 그룹화

chg_e_stored_total = 0
chg_energy_total = 0

for seg_id, group in chg_groups:
    group = group.sort_values('time')#데이터 시간 순 정렬

    soc_start = group['soc'].iloc[0] / 100
    soc_end = group['soc'].iloc[-1] / 100

    #if abs(soc_end - soc_start) < 0.001:
        #continue  # SOC 변화 거의 없으면 건너뜀

    e_start = compute_estored(soc_start)
    e_end = compute_estored(soc_end) #ocv-soc곡선으로 e_stored 계산

    # Wh 단위 충전 에너지 계산
    e_charging = abs((group['pack_volt'] * group['pack_current'] * group['delta_sec']) / 3600).sum()

    chg_e_stored_total += max(e_end - e_start, 0) #만약에 e_stored가 음수라면, 0을 반환.
                                                  #즉 양수인, 충전이 되는 것만 반환
    chg_energy_total += e_charging

# 4. 충전 효율 계산
if chg_energy_total > 0:
    e_chg = (chg_e_stored_total / chg_energy_total) * 100
    print(f"[충전 효율] e_chg = {e_chg:.2f}%")
    print(f" - 저장 에너지 합: {chg_e_stored_total:.2f} Wh")
    print(f" - 실제 충전 에너지 합: {chg_energy_total:.2f} Wh")
else:
    print("⚠️ 충전 에너지 합이 0입니다. e_chg 계산 불가")







# 충전 구간 데이터만 필터링
charging_df = df[(df['chrg_cable_conn'] == 1) & (df['speed'] == 0) & (df['pack_current'] < 0)].copy()

# delta_sec이 초 단위 숫자이므로 .dt.total_seconds() 제거
charging_df['time_diff'] = charging_df['delta_sec'].diff().fillna(0)

# time_diff가 60초 초과면 새로운 그룹으로 나누기
charging_df['group_id'] = (charging_df['time_diff'] > 20).cumsum()

# 그룹별로 나누기
charging_groups = [group for _, group in charging_df.groupby('group_id')]



for seg_id, group in chg_groups:
    soc_start = group['soc'].iloc[0]
    soc_end = group['soc'].iloc[-1]
    print(f"충전 구간 {int(seg_id)}: SOC {soc_start:.2f}% → {soc_end:.2f}% ({soc_end - soc_start:.2f}%)")


##################################################################################################################


for i, group in enumerate(charging_groups):
    soc_start = group['soc'].iloc[0] / 100
    soc_end = group['soc'].iloc[-1] / 100

    E_stored_start = compute_estored(soc_start)
    E_stored_end = compute_estored(soc_end)
    E_stored_delta = E_stored_end - E_stored_start

    E_charging_sum = (group['pack_volt'] * group['pack_current'].abs() * group['delta_sec'] / 3600).sum()

    charging_only = group[group['pack_current'] < -1]  # 예: 충전임을 나타내는 조건

    E_charging_sum = (charging_only['pack_volt'] * charging_only['pack_current'].abs() * charging_only[
        'delta_sec'] / 3600).sum()

    print(f"충전 구간 {i + 1}: SOC {soc_start * 100:.2f}% → {soc_end * 100:.2f}% (ΔSOC {100 * (soc_end - soc_start):.2f}%)")
    print(f" - 저장 에너지 변화: {E_stored_delta:.2f} Wh")
    print(f" - 실제 충전 에너지 합: {E_charging_sum:.2f} Wh\n")



group['instant_energy'] = -group['pack_volt'] * group['pack_current'] * group['delta_sec'] / 3600
print(group[['time', 'pack_volt', 'pack_current', 'delta_sec', 'instant_energy']].head(10))
print(f"합계: {group['instant_energy'].sum():.2f} Wh")









print(group['time'].iloc[0], group['time'].iloc[-1])
print("총 충전 시간 (초):", (group['time'].iloc[-1] - group['time'].iloc[0]).total_seconds())



print(group['delta_sec'].describe())
print(group[['time', 'delta_sec']].head(20))


print(f"SOC 변화: {group['soc'].iloc[-1] - group['soc'].iloc[0]:.2f}%")



print(f"충전 구간 {i+1}")
print(f" - 충전 전류 평균: {charging_only['pack_current'].mean():.2f} A")
print(f" - 충전 지속 시간: {charging_only['delta_sec'].sum():.0f} 초")
print(f" - 실제 충전 에너지 합: {E_charging_sum:.2f} Wh\n")



import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -- (기존 코드 유지, bms_df, df, chg_groups 등 변수 준비 끝난 상태라고 가정) --

# 외기온도 컬럼명 (예: ext_temp) 맞게 바꿔주세요
ext_temp_col = 'ext_temp'
if ext_temp_col not in bms_df.columns:
    raise ValueError(f"외기온도 컬럼 '{ext_temp_col}'가 bms_df에 없습니다.")

# 1) 월별로 외기온도 평균 계산 (x축용)
bms_df['year_month'] = bms_df['time'].dt.to_period('M').dt.to_timestamp()
temp_monthly = bms_df.groupby('year_month')[ext_temp_col].mean()

# 2) 효율 값 (기존 계산된 값들)
# e_chg = (chg_e_stored_total / chg_energy_total) * 100
# efficiency1 = E_trip_net / (E_charging + e_stored_diff) * 100
# efficiency2 = (E_trip_net + e_stored_1) / (E_charging + e_stored_0) * 100

# 효율을 월별 시리즈로 만들려면 월별로 계산 반복 필요하나, 단일값이라면 그냥 일정한 y값으로 선 표시
# 만약 월별로 여러 충전 구간이 있다면 충전 구간별 e_chg 계산 후 월별 평균내야 하지만,
# 현재는 단일값이라 가정하고 모든 월에 같은 값 선으로 표시

months = temp_monthly.index

# 3) 충전 구간 시간 범위 추출 (x축 충전 구간 표시)
# chg_groups는 충전 구간별 그룹
charging_intervals = []
for seg_id, group in chg_groups:
    start_time = group['time'].iloc[0]
    end_time = group['time'].iloc[-1]
    charging_intervals.append((start_time, end_time))

# --- 시각화 시작 ---
fig, ax1 = plt.subplots(figsize=(15,7))

# 왼쪽 y축: 외기온도 월별 평균 선그래프
ax1.plot(temp_monthly.index, temp_monthly.values, color='tab:blue', label='외기온도 (°C)')
ax1.set_xlabel('시간 (월별)')
ax1.set_ylabel('외기온도 (°C)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# x축 월별 포맷 설정
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xticks(rotation=45)

# 오른쪽 y축: 효율 %
ax2 = ax1.twinx()
ax2.set_ylabel('효율 (%)')

# 효율 선 그리기
ax2.plot(months, [e_chg]*len(months), label='e_chg', color='tab:red', linestyle='-')
ax2.plot(months, [efficiency1]*len(months), label='efficiency1', color='tab:green', linestyle='--')
ax2.plot(months, [efficiency2]*len(months), label='efficiency2', color='tab:orange', linestyle=':')

ax2.tick_params(axis='y')

# 충전 구간을 x축 아래에 색깔 영역으로 표시
for start, end in charging_intervals:
    ax1.axvspan(start, end, color='gray', alpha=0.3)

# 범례
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('월별 외기온도 및 충전 효율 비교')
plt.tight_layout()
plt.show()






