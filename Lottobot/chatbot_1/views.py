from rest_framework.views import APIView
from rest_framework.response import Response
import chromadb
import random
from django.conf import settings
import openai


# OpenAI API 키 설정 (settings.py에 저장해둔 값을 활용)
openai.api_key = settings.OPENAI_API_KEY


# ChromaDB 클라이언트 연결
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection = chroma_client.get_or_create_collection(name="lotto_numbers")


class ChatbotView(APIView):
    """
    사용자의 메시지를 받아, ChromaDB의 로또 데이터와 OpenAI LLM을 활용해 답변을 생성하는 챗봇 API
    """

    def post(self, request, *args, **kwargs):
        # 사용자 메시지 받기
        user_message = request.data.get("message", "").strip()
        if not user_message:
            return Response({"error": "메시지가 비어있습니다."}, status=400)

        # ChromaDB로부터 데이터 조회
        all_data = collection.get(include=["metadatas"])
        if not all_data["metadatas"]:
            return Response({"error": "데이터가 없습니다."}, status=404)

        # 최대값 뽑아내기
        max_count = max([data["count"] for data in all_data["metadatas"]])

        # 백터 차원 맞춰주기
        query_vector = [max_count] * 5

        # 결과 뽑아내기
        results = collection.query(query_embeddings=query_vector, n_results=12)
        recommended_top_numbers = [
            {"number": res["number"], "count": res["count"]}
            for res in results["metadatas"][0]
        ]
        # 리스트를 랜덤하게 섞은 후 상위 6개 번호 선택
        random.shuffle(recommended_top_numbers)

        top_numbers = recommended_top_numbers[:6]
        top_numbers_str = ", ".join(str(item["number"]) for item in top_numbers)

        # 프롬프트
        # 사용자의 질문과 로또 데이터(상위 번호)를 함께 포함하여 OpenAI에게 요청합니다.
        prompt = f"""사용자의 질문: {user_message}
10년간 로또 당첨 데이터에 따르면, 가장 당첨 확률이 높았던 번호는 다음과 같습니다: {top_numbers_str}.
이 번호를 추천하는 이유를 친절한 말투로 간결하게 알려줘. 
"""

        # OpenAI API 호출 (ChatCompletion 방식)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 로또 챗봇입니다. 사용자의 질문에 대해 로또 데이터에 기반한 정보를 제공하세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            answer = response.choices[0].message["content"].strip()
        except Exception as e:
            return Response({"error": f"OpenAI API 호출 실패: {str(e)}"}, status=500)

        # 5. 응답 반환
        return Response({"answer": answer})


# -------------------------------------삭제예정-------------------------------------
# class TopLottoNumbersView(APIView):
#     """
#     ChromaDB에서 가장 출현 빈도가 높은 6개 로또 번호를 반환하는 API (CBV)
#     """

#     def get(self, request, *args, **kwargs):

#         all_data = collection.get(include=["metadatas", "embeddings"])
#         if not all_data["metadatas"]:  # 데이터가 없을 경우 예외 처리
#             return Response({"message": "No data available"}, status=404)

#         # 출현 빈도 최댓값 찾기
#         max_count = max([data["count"] for data in all_data["metadatas"]])

#         query_vector = [max_count] * 5

#         # 최대값과 가장 가까운값 찾기
#         # 출현 빈도가 같은 숫자가 있을 경우, 검색 결과가 달라질 수 있음.
#         results = collection.query(query_embeddings=query_vector, n_results=12)

#         # 결과 -> 메타데이터에서 숫자와 빈도를 가져옴
#         # 실제 서비스 때는 숫자만 가져오도록 변경예정. 일단 확인차 빈도도 가져오기.
#         recommended_top_numbers = [
#             {"number": res["number"], "count": res["count"]}
#             for res in results["metadatas"][0]  # 백터디비(2차원)이라 [0] 사용
#         ]
#         # 리스트를 랜덤하게 섞음
#         random.shuffle(recommended_top_numbers)

#         # 6개만 선택
#         return Response({"top_lotto_numbers": recommended_top_numbers[:6]})
