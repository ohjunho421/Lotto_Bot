import pandas as pd
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(open_api_key=OPENAI_API_KEY)

CHROMA_DB_PATH = "./chroma_db"


def save_lotto_data_to_chroma():
    # 데이터 로드
    df = pd.read_excel("lotto.xlsx")
    # 전처리
    df = df.drop([0,1])
    df = df.drop("회차별 추첨결과", axis=1)
    # 한글이면 성능떨어짐 -> 나중에 영문으로바꿔야함
    df.columns = ['draw_no', 'draw_date', 'winner_1st', 'prize_1st', 'winner_2nd', 'prize_2nd', 'winner_3rd', 'prize_3rd', 'winner_4th', 'prize_4th', 'winner_5th', 'prize_5th', 'num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'bonus_num']
    df = df.astype({col: int for col in df.select_dtypes(include=["float64"]).columns})
    
    # 데이터 문자열로 변환
    df["text"] = df.apply(lambda row: f"{row['회차']}회차 당첨번호: {', '.join(map(str, row[['당첨번호1', '당첨번호2', '당첨번호3', '당첨번호4', '당첨번호5', '당첨번호6']]))}", axis=1)
    
    #벡터디비 저장
    vector_store = Chroma.from_texts(df["text"].tolist(), embedding_model, persist_directory=CHROMA_DB_PATH)
    vector_store.persist()
    print("로또 데이터 벡터DB 저장 완료!")
    

def retrieve_relevant_lotto_info(user_query, top_n=3):
    # 벡터DB에서 사용자 질문과 유사한 데이터 검색
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    docs = vector_store.similarity_search(user_query, k=top_n)
    return [doc.page_content for doc in docs]
