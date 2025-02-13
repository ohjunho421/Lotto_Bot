import chromadb
import pandas as pd

# 크로마디비에 영구저장하기 Persistent ,
# 컬렉션 생성 (컬렉션(Collection) = 벡터 데이터를 저장하는 테이블 같은 개념)
chroma_client = chromadb.PersistentClient(path="./chromadb_data")


try:
    chroma_client.delete_collection(name="lotto_numbers")  # 기존 컬렉션 삭제
except Exception:
    pass  # 컬렉션이 없으면 그냥 넘어감

collection = chroma_client.create_collection(name="lotto_numbers")  # 새로 생성


# 판다스로 데이터 로드
df = pd.read_csv("chatbot_1/Lotto_Data.csv", encoding="euc-kr")

# 데이터 계산
number_columns = ["num_1", "num_2", "num_3", "num_4", "num_5", "num_6", "bonus_num"]
number_counts = df[number_columns].stack().value_counts().reset_index()
number_counts.columns = ["number", "count"]

# 스크립트 실행때마다 중복저장 되는거 방지 (컬렉션 비우기)
# -> 이거 안하면 스크립트 실행때마다 데이터가 중복으로 쌓임.
# 우린 크롤링도 할거라 무조건 매주 실행하기 때문에 중복방지를 위해 지워야함.
# collection.delete(where={})


# 계산된 데이터 컬렉션(벡터디비)에 저장
"""
ChromaDB는 데이터를 ID, 벡터(embedding), 메타데이터(metadata) 3가지 형태로 저장
여기서 id는 각 숫자, 임베딩은 출현빈도, 메타데이타는 추가정보를 넣을 수 있음.
추후에 추가정보에 다른 정보도 추가하면 좋을 것 같음. 예를들어, 출연한 회차 정보 등.
"""

for i, row in number_counts.iterrows():

    # 번호가 등장한 회차 찾기
    draws = list(df[df[number_columns].eq(row["number"])].draw_no)
    valid_draws = [str(x) for x in draws if pd.notna(x)]
    # 5차원 임베딩 벡터 생성
    embedding = [float(row["count"])] * 5

    collection.add(
        ids=[str(row["number"])],
        embeddings=[embedding],
        metadatas=[
            {
                "number": int(row["number"]),
                "count": int(row["count"]),
            }
        ],
    )

"""
제이슨 형태로 저장
{
  "id": "21",
  "embedding": [89],
  "metadata": {"number": 21, "count": 89},
}
"""
print("ChromaDB 데이터 저장 완!")


data = collection.get(include=["embeddings", "metadatas"])
print("저장된 데이터:", data)
