import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# 데이터 로드
df = pd.read_excel("data/lottobot.xlsx")
# 전처리
df = df.drop([0,1])
df = df.drop("회차별 추첨결과", axis=1)
df.columns = ['draw_no', 'draw_date', 'winner_1st', 'prize_1st', 'winner_2nd', 'prize_2nd', 'winner_3rd', 'prize_3rd', 'winner_4th', 'prize_4th', 'winner_5th', 'prize_5th', 'num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'bonus_num']
df = df.astype({col: int for col in df.select_dtypes(include=["float64"]).columns})
# df['draw_date'] = pd.to_datetime(df['draw_date'], format='%Y.%m.%d')
df['draw_date'] = df['draw_date'].str.replace('.', '-')
for column in ['prize_1st', 'prize_2nd', 'prize_3rd', 'prize_4th', 'prize_5th']:
    df[column] = df[column].str.replace(',', '').str.replace('원', '')

# 필요한 컬럼 선택 후 문자열 변환
df["processed_text"] = df.apply(
    lambda row: f"{row['draw_no']}회차는 {row['draw_date']}에 진행되었고, 당첨숫자는 {row['num_1']}, {row['num_2']}, {row['num_3']}, {row['num_4']}, {row['num_5']}, {row['num_6']}이며, 보너스 숫자는 {row['bonus_num']}이었습니다.",
    axis=1
)

# 전처리된 텍스트 리스트로 변환
lotto_docs = df["processed_text"].tolist()

print(lotto_docs[0])
print(len(lotto_docs[0]))




