import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# -----------------------
# 파일 불러오기
# -----------------------
file_info = {
    r"E:\EV6_1year\06\bms_merged_06.csv": "veh06",
    r"E:\EV6_1year\49\bms_merged_49.csv": "veh49",
    # ...
}

dfs = []
for path, vehicle_name in file_info.items():
    if os.path.exists(path):
        df_tmp = pd.read_csv(path)
        df_tmp['vehicle'] = vehicle_name
        dfs.append(df_tmp)
    else:
        print(f"⚠️ 파일 없음: {path}")

if not dfs:
    raise FileNotFoundError("로드할 CSV 파일이 없습니다.")

bms_df = pd.concat(dfs, ignore_index=True)

# -----------------------
# OCV-SOC 데이터 로드
# -----------------------
ocv_file = r"C:\Users\OWNER\Desktop\EV6_2301\NE_Cell_Characterization_performance.xlsx"
if not os.path.exists(ocv_file):
    raise FileNotFoundError(f"OCV 파일을 찾을 수 없습니다: {ocv_file}")

try:
    ocv_raw = pd.read_excel(ocv_file, sheet_name='SOC-OCV')
    match_idx = ocv_raw[ocv_raw.iloc[:, 6] == 'SOC (%)'].index
    if len(match_idx) == 0:
        raise ValueError("'SOC (%)' 행을 찾지 못했습니다.")
    start_row = match_idx[0] + 1
except Exception as e:
    raise RuntimeError(f"OCV 데이터 로드 오류: {e}")

soc_ocv_data = ocv_raw.iloc[start_row:, [6, 9]]
soc_ocv_data.columns = ['SOC', 'OCV']
soc_ocv_data = soc_ocv_data.dropna().astype(float)

# SOC-OCV Lookup
soc_vals = soc_ocv_data['SOC'].values / 100
ocv_vals = soc_ocv_data['OCV'].values
order = np.argsort(soc_vals)
soc_vals, ocv_vals = soc_vals[order], ocv_vals[order]

Qmax = 56.47
soc_grid = np.linspace(0, 1, 1001)
ocv_interp = interp1d(soc_vals, ocv_vals, kind='linear', fill_value='extrapolate')(soc_grid)
estored_grid = Qmax * cumulative_trapezoid(ocv_interp, soc_grid, initial=0) * 192 * 2  # Wh

def fast_estored(soc):
    return np.interp(soc, soc_grid, estored_grid)

# -----------------------
# 모듈 온도 평균
# -----------------------
def parse_mod_temp_list(val):
    try:
        nums = list(map(float, str(val).split(',')))
        return np.mean(nums)
    except:
        return np.nan

if 'mod_temp_list' in bms_df.columns:
    bms_df['mod_temp_avg'] = bms_df['mod_temp_list'].apply(parse_mod_temp_list)
else:
    bms_df['mod_temp_avg'] = np.nan

# -----------------------
# 시간/세그먼트 처리
# -----------------------
if 'time' not in bms_df.columns:
    raise KeyError("'time' 컬럼이 필요합니다.")

bms_df['time'] = pd.to_datetime(bms_df['time'], errors='coerce')
bms_df = bms_df.sort_values('time').reset_index(drop=True)
bms_df['delta_sec_raw'] = bms_df['time'].diff().dt.total_seconds().fillna(0)

gap_indices = bms_df.index[bms_df['delta_sec_raw'] > 60].tolist()
segment_id = 0
segments = []
for i in range(len(bms_df)):
    segments.append(segment_id)
    if i in gap_indices:
        segment_id += 1
bms_df['segment_id'] = segments
bms_df['delta_sec'] = bms_df.groupby('segment_id')['time'].diff().dt.total_seconds().fillna(0)
bms_df['valid'] = bms_df['delta_sec'] <= 10

# -----------------------
# 조건 정의 (벡터화)
# -----------------------
calc_energy = lambda v, c, t: (v * c * t) / 3600  # Wh

cond_drive = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 0) & (bms_df['speed'] > 0) & (bms_df['pack_current'] > 0)
cond_idle = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 0) & (bms_df['speed'] == 0) & (bms_df['pack_current'] > 0)
cond_regen = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 0) & (bms_df['pack_current'] < 0)
cond_chg = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 1) & (bms_df['speed'] == 0) & (bms_df['pack_current'] < 0)
cond_chg_idle = bms_df['valid'] & (bms_df['chrg_cable_conn'] == 1) & (bms_df['speed'] == 0) & (bms_df['pack_current'] > 0)

bms_df['E_drive'] = 0.0
bms_df.loc[cond_drive, 'E_drive'] = calc_energy(bms_df.loc[cond_drive, 'pack_volt'],
                                                bms_df.loc[cond_drive, 'pack_current'],
                                                bms_df.loc[cond_drive, 'delta_sec'])

bms_df['E_idle'] = 0.0
bms_df.loc[cond_idle, 'E_idle'] = calc_energy(bms_df.loc[cond_idle, 'pack_volt'],
                                              bms_df.loc[cond_idle, 'pack_current'],
                                              bms_df.loc[cond_idle, 'delta_sec'])

bms_df['E_regen'] = 0.0
bms_df.loc[cond_regen, 'E_regen'] = calc_energy(bms_df.loc[cond_regen, 'pack_volt'],
                                                -bms_df.loc[cond_regen, 'pack_current'],
                                                bms_df.loc[cond_regen, 'delta_sec'])

bms_df['E_chg_real'] = 0.0
bms_df.loc[cond_chg, 'E_chg_real'] = calc_energy(bms_df.loc[cond_chg, 'pack_volt'],
                                                 -bms_df.loc[cond_chg, 'pack_current'],
                                                 bms_df.loc[cond_chg, 'delta_sec'])

bms_df['E_chg_idle'] = 0.0
bms_df.loc[cond_chg_idle, 'E_chg_idle'] = calc_energy(bms_df.loc[cond_chg_idle, 'pack_volt'],
                                                      bms_df.loc[cond_chg_idle, 'pack_current'],
                                                      bms_df.loc[cond_chg_idle, 'delta_sec'])

# -----------------------
# condition 할당 (apply 제거, 벡터화)
# -----------------------
bms_df['condition'] = np.where(
    bms_df['chrg_cable_conn'] == 1, 'charge',
    np.where(
        (bms_df['speed'] == 0) & (bms_df['chrg_cable_conn'] == 0) & (bms_df['delta_sec'] > 600),
        'rest',
        'drive'
    )
)

# -----------------------
# trip_id별 E_stored_diff 계산
# -----------------------
if 'trip_id' in bms_df.columns and 'soc' in bms_df.columns:
    estored_df = (
        bms_df[bms_df['valid']]
        .groupby('trip_id')['soc']
        .agg(lambda x: fast_estored(x.iloc[0]/100) - fast_estored(x.iloc[-1]/100))
        .reset_index(name='E_stored_diff')
    )
    bms_df = bms_df.merge(estored_df, on='trip_id', how='left')
else:
    bms_df['E_stored_diff'] = np.nan

# -----------------------
# 평균 모듈온도 계산
# -----------------------
mean_modtemp = bms_df.groupby('condition')['mod_temp_avg'].mean()
bms_df['modtemp_drive'] = mean_modtemp.get('drive', np.nan)
bms_df['modtemp_rest'] = mean_modtemp.get('rest', np.nan)
bms_df['modtemp_charge'] = mean_modtemp.get('charge', np.nan)

# -----------------------
# 효율 계산
# -----------------------
bms_df['E_trip_net'] = (bms_df['E_drive'] + bms_df['E_idle']) - bms_df['E_regen']
bms_df['E_charging'] = bms_df['E_chg_real'] - bms_df['E_chg_idle']
bms_df['efficiency1'] = bms_df['E_trip_net'] / (bms_df['E_charging'] + bms_df['E_stored_diff']) * 100

# -----------------------
# 그룹화 및 회귀분석
# -----------------------
# -----------------------
# 그룹화 및 회귀분석
# -----------------------
if 'trip_id' in bms_df.columns:
    grouped = bms_df.groupby('trip_id').agg({
        'efficiency1': 'mean',
        'modtemp_drive': 'mean',
        'modtemp_rest': 'mean',
        'modtemp_charge': 'mean'
    }).dropna()
else:
    grouped = pd.DataFrame({
        'efficiency1': [bms_df['efficiency1'].mean()],
        'modtemp_drive': [bms_df['modtemp_drive'].mean()],
        'modtemp_rest': [bms_df['modtemp_rest'].mean()],
        'modtemp_charge': [bms_df['modtemp_charge'].mean()]
    })

# 데이터 개수 확인 후 회귀 수행
if len(grouped) >= 2:
    X = sm.add_constant(grouped[['modtemp_drive', 'modtemp_rest', 'modtemp_charge']])
    y = grouped['efficiency1']
    model = sm.OLS(y, X).fit()
    print(model.summary())
else:
    print("⚠️ 회귀분석 스킵: trip 데이터가 2개 미만이어서 회귀를 수행할 수 없습니다.")
    print("현재 grouped 데이터:")
    print(grouped)


X = sm.add_constant(grouped[['modtemp_drive', 'modtemp_rest', 'modtemp_charge']])
y = grouped['efficiency1']

model = sm.OLS(y, X).fit()
print(model.summary())
