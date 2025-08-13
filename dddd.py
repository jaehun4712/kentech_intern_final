   from scipy.stats import pearsonr
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. e_trip 기준 하위 20% trip 필터링
    threshold = df_trip_summary['e_trip'].quantile(0.2)
    low_eff_df = df_trip_summary[df_trip_summary['e_trip'] <= threshold]

    # 2. 상관계수 계산 (피어슨)
    r_ext, p_ext = pearsonr(low_eff_df['e_trip'], low_eff_df['ext_temp_avg'])
    r_mod, p_mod = pearsonr(low_eff_df['e_trip'], low_eff_df['mod_temp_avg'])

    print(f"[하위 20% 효율 trip 기준]")
    print(f"외기온도 vs 효율: r = {r_ext:.2f}, p = {p_ext:.3f}")
    print(f"모듈온도 vs 효율: r = {r_mod:.2f}, p = {p_mod:.3f}")

    # 3. 시각화
    sns.lmplot(data=low_eff_df, x='ext_temp_avg', y='e_trip')
    plt.title("하위 20%: 외기온도 vs e_trip 효율")
    plt.show()

    sns.lmplot(data=low_eff_df, x='mod_temp_avg', y='e_trip')
    plt.title("하위 20%: 모듈온도 vs e_trip 효율")
    plt.show()

# 하위 10% & 상위 10% 필터링
low_10 = df_trip_summary[df_trip_summary['e_trip'] <= df_trip_summary['e_trip'].quantile(0.10)]
high_10 = df_trip_summary[df_trip_summary['e_trip'] >= df_trip_summary['e_trip'].quantile(0.90)]

# 상관계수 계산
r_low_ext, p_low_ext = pearsonr(low_10['e_trip'], low_10['ext_temp_avg'])
r_high_ext, p_high_ext = pearsonr(high_10['e_trip'], high_10['ext_temp_avg'])

print("[하위 10% 효율 trip]")
print(f"외기온도 vs 효율: r = {r_low_ext:.2f}, p = {p_low_ext:.3f}")
print("[상위 10% 효율 trip]")
print(f"외기온도 vs 효율: r = {r_high_ext:.2f}, p = {p_high_ext:.3f}")


# 온도 구간 라벨링
def temp_group(temp):
    if temp <= 0:
        return "0℃ 이하"
    elif temp <= 10:
        return "0~10℃"
    else:
        return "10℃ 초과"

df_trip_summary['temp_group'] = df_trip_summary['ext_temp_avg'].apply(temp_group)

# 각 온도 구간의 e_trip_eff 평균 시각화
sns.boxplot(data=df_trip_summary, x='temp_group', y='e_trip')
plt.title("외기온도 구간별 e_trip 효율 분포")
plt.xlabel("외기온도 구간")
plt.ylabel("e_trip 효율")
plt.show()

# 각 구간별 평균값 출력
grouped = df_trip_summary.groupby('temp_group')['e_trip'].describe()
print(grouped[['mean', 'std', 'count']])