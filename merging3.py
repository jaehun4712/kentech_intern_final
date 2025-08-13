import pandas as pd
import glob

# 1. 병합할 폴더 경로 설정
folder_path = r"E:\EV6_1year\107"

# 2. 해당 폴더 내 bms_*.csv 파일들만 모두 가져오기
file_list = glob.glob(folder_path + r"\bms_*.csv")

print("📂 총 파일 수:", len(file_list))  # 12개인지 확인

# 3. 모든 파일 읽어서 리스트에 저장
df_list = [pd.read_csv(file) for file in file_list]

# 4. 병합
df_merged = pd.concat(df_list, ignore_index=True)

# 5. 날짜 컬럼 확인 후 정렬
print(df_merged.columns)  # 날짜 컬럼명 확인

# 예: 'timestamp' 컬럼이 날짜일 경우 (이름에 맞게 수정!)
df_merged['time'] = pd.to_datetime(df_merged['time'])
df_merged = df_merged.sort_values(by='time')

# 6. 병합 결과 저장
save_path = folder_path + r"\bms_merged_sorted.csv"
df_merged.to_csv(save_path, index=False)

print("✅ 병합 및 정렬 완료! 저장 위치:", save_path)

