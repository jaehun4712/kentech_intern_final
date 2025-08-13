import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 파일 불러오기 및 설정
# -----------------------
file_info = {
    r"E:\EV6_1year\06\bms_merged_06.csv": "veh06",
    r"E:\EV6_1year\49\bms_merged_49.csv": "veh49",
    r"E:\EV6_1year\51\bms_merged_51.csv": "veh51",
    #r"E:\EV6_1year\53\bms_merged_53.csv": "veh53",
    #r"E:\EV6_1year\54\bms_merged_54.csv": "veh54",
    #r"E:\EV6_1year\59\bms_merged_59.csv": "veh59",
    #r"E:\EV6_1year\57\bms_merged_57.csv": "veh57"

}

dfs = []
for path, vehicle_name in file_info.items():
    df_tmp = pd.read_csv(path)
    df_tmp['vehicle'] = vehicle_name
    dfs.append(df_tmp)

bms_df = pd.concat(dfs, ignore_index=True)

ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------
# 모듈 온도 평균 계산 함수
# -----------------------
def parse_mod_temp_list(val):
    try:
        nums = list(map(float, str(val).split(',')))
        return np.mean(nums)
    except:
        return np.nan

bms_df['mod_temp_avg'] = bms_df['mod_temp_list'].apply(parse_mod_temp_list)

# -----------------------
# 시간 및 세그먼트 처리
# -----------------------
bms_df['time'] = pd.to_datetime(bms_df['time'], errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)
bms_df['week'] = bms_df['time'].dt.to_period('W').dt.start_time
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)

# 세그먼트 분리 (시간 간격 > 60초 구간에서 세그먼트 변경)
gap_indices = bms_df.index[bms_df['delta_sec_raw'] > 60].tolist()
segment_id = 0
segments = []
for i in range(len(bms_df)):
    segments.append(segment_id)
    if i in gap_indices:
        segment_id += 1
bms_df['segment_id'] = segments
bms_df['delta_sec'] = bms_df.groupby('segment_id')['time'].diff().dt.total_seconds().fillna(0)

# 유효 데이터 조건: delta_sec <= 10초
bms_df['valid'] = bms_df['delta_sec'] <= 10

# -----------------------
# OCV-SOC 적분 lookup 테이블 생성
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
soc_grid = np.linspace(0, 1, 1001)
ocv_interp = interp1d(soc_vals, ocv_vals, kind='linear', fill_value='extrapolate')(soc_grid)
estored_grid = Qmax * cumulative_trapezoid(ocv_interp, soc_grid, initial=0) * 192 * 2  # Wh 단위

def fast_estored(soc):
    return np.interp(soc, soc_grid, estored_grid)

# -----------------------
# 에너지 계산 (벡터화)
# -----------------------
def calc_energy(voltage, current, delta_sec):
    return (voltage * current * delta_sec) / 3600  # Wh

# 조건 정의
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
# SOC 기반 에너지 변화 계산 함수
# -----------------------
def soc_diff_energy(df):
    soc_series = df[df['valid']]['soc'].dropna()
    if len(soc_series) < 2:
        return np.nan, np.nan, np.nan
    soc0 = soc_series.iloc[0] / 100
    soc1 = soc_series.iloc[-1] / 100
    return fast_estored(soc0), fast_estored(soc1), fast_estored(soc0) - fast_estored(soc1)

# -----------------------
# 주차별 집계 및 계산
# -----------------------
weekly_df = bms_df.groupby(['vehicle', 'week']).apply(
    lambda g: pd.Series({
        'E_drive': g['E_drive'].sum(),
        'E_idle': g['E_idle'].sum(),
        'E_regen': g['E_regen'].sum(),
        'E_chg_real': g['E_chg_real'].sum(),
        'E_chg_idle': g['E_chg_idle'].sum(),
        'E_stored_0': soc_diff_energy(g)[0],
        'E_stored_1': soc_diff_energy(g)[1],
        'E_stored_diff': soc_diff_energy(g)[2],
        'mod_temp_avg': g['mod_temp_avg'].mean(),
        'ext_temp_avg': g['ext_temp'].mean()
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
# 상관계수 계산
# -----------------------
r1, p1 = pearsonr(weekly_df['mod_temp_avg'], weekly_df['efficiency1'])
r2, p2 = pearsonr(weekly_df['mod_temp_avg'], weekly_df['efficiency2'])

# -----------------------
# 시각화
# -----------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
sns.scatterplot(data=weekly_df, x='mod_temp_avg', y='efficiency1', hue='vehicle', ax=axs[0])
sns.regplot(data=weekly_df, x='mod_temp_avg', y='efficiency1', scatter=False, ax=axs[0], color='gray')
axs[0].set_title(f'온도 vs 효율 e1')

sns.scatterplot(data=weekly_df, x='mod_temp_avg', y='efficiency2', hue='vehicle', ax=axs[1])
sns.regplot(data=weekly_df, x='mod_temp_avg', y='efficiency2', scatter=False, ax=axs[1], color='gray')
axs[1].set_title(f'온도 vs 효율 e2')

for ax in axs:
    ax.set_xlabel('모듈온도(°C)')
    ax.set_ylabel('효율(%)')
    ax.grid(True)

plt.tight_layout()
plt.show()

#####################################################

from scipy.stats import linregress

# x, y 데이터 준비
x = weekly_df['mod_temp_avg']
y = weekly_df['efficiency1']

# 선형 회귀 분석
result = linregress(x, y)

# 회귀선 그리기
plt.figure(figsize=(8,6))
sns.scatterplot(x=x, y=y, color='tab:green', label='Data Points')
sns.lineplot(x=x, y=result.intercept + result.slope * x, color='tab:blue', label='Fit Line')

# 통계값 텍스트 추가
r_value = result.rvalue
p_value = result.pvalue
r_squared = r_value**2

plt.text(0.05, 0.95, f"r = {r_value:.3f}\np = {p_value:.3e}\nR² = {r_squared:.3f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

plt.xlabel('모듈온도(°C)')
plt.ylabel('효율 e1 (%)')
plt.title('모듈온도 vs 효율 e1 (전체 차량 데이터)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# efficiency2 데이터
x2 = weekly_df['mod_temp_avg']
y2 = weekly_df['efficiency2']

# 선형 회귀 분석
result2 = linregress(x2, y2)

# 그래프 그리기
plt.figure(figsize=(8,6))
sns.scatterplot(x=x2, y=y2, color='tab:orange', label='Data Points')
sns.lineplot(x=x2, y=result2.intercept + result2.slope * x2, color='tab:red', label='Fit Line')

# 통계값 텍스트 추가
r_value2 = result2.rvalue
p_value2 = result2.pvalue
r_squared2 = r_value2**2

plt.text(0.05, 0.95, f"r = {r_value2:.3f}\np = {p_value2:.3e}\nR² = {r_squared2:.3f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

plt.xlabel('모듈온도(°C)')
plt.ylabel('효율 e2 (%)')
plt.title('모듈온도 vs 효율 e2 (전체 차량 데이터)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

###########################################################
import statsmodels.api as sm

# 독립 변수 (모듈온도, 외기온도)
X = weekly_df[['mod_temp_avg', 'ext_temp_avg']]

# 종속 변수 (효율)
y = weekly_df['efficiency1']

# 상수항 추가 (절편)
X = sm.add_constant(X)

# 모델 적합
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())



import matplotlib.pyplot as plt

# 예측값 계산
y_pred = model.predict(X)

plt.scatter(y, y_pred)
plt.xlabel('실제 효율')
plt.ylabel('예측 효율')
plt.title('다중 선형 회귀: 실제 vs 예측')

# R-squared 텍스트 출력
plt.text(min(y), max(y), f'R² = {model.rsquared:.3f}', fontsize=12)

plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')  # 45도 선
plt.show()
