from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import Chatbot
from .serializers import ChatbotSerializer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

import pandas as pd
import openai, os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 
load_dotenv()
# API Key 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

persist_directory = "Lotto_history_vector_store"
if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
else:
    # 데이터 로드 및 전처리
    df = pd.read_excel("data/lottobot.xlsx")
    df = df.drop([0,1])
    df = df.drop("회차별 추첨결과", axis=1)
    df.columns = ['draw_no', 'draw_date', 'winner_1st', 'prize_1st', 'winner_2nd', 'prize_2nd', 'winner_3rd', 'prize_3rd', 'winner_4th', 'prize_4th', 'winner_5th', 'prize_5th', 'num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'bonus_num']
    df = df.astype({col: int for col in df.select_dtypes(include=["float64"]).columns})
    df['draw_date'] = df['draw_date'].str.replace('.', '-')
    for column in ['prize_1st', 'prize_2nd', 'prize_3rd', 'prize_4th', 'prize_5th']:
        df[column] = df[column].str.replace(',', '').str.replace('원', '')

    # 필요한 컬럼 선택 후 문자열 변환
    df["processed_text"] = df.apply(
        lambda row: f"로또 {row['draw_no']}회차는 {row['draw_date']}에 진행되었고, 당첨숫자는 {row['num_1']}, {row['num_2']}, {row['num_3']}, {row['num_4']}, {row['num_5']}, {row['num_6']}이며, 보너스 숫자는 {row['bonus_num']}이었습니다.",
        axis=1
    )

    # 전처리된 텍스트 리스트로 변환
    lotto_docs = df["processed_text"].tolist()
    docs = [Document(page_content=text) for text in lotto_docs]

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)  # 문맥이 이어진정보가아닌, 단순 72글자 짜리 당첨정보라 청킹을 해야하나 의문 -> 해보고안해보고 성능비교
    # splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

# 벡터 스토어에서 검색을 위한 retriever 생성
retriever = vector_store.as_retriever()


# 프롬프트 템플릿 설정 -> 사용자가 전략 추가하게하면 전략별로 prompt1, prompt2, prompt3, ... 이런식으로 추가하면 어떨까?? 생각중
# llm 모델이 유사도기반으로 추론을 하지 패턴분석 및 통계적 인사이트를 가지기엔 아직 부족함이 많음
# 준호님이 ML 돌린거 or 직접 패턴분석 or 통계적 분석후 결과를 프롬프트에 전달해서 추천해달라고 하는게 베스트
prompt = ChatPromptTemplate.from_template(
    """
당신은 예상 로또당첨번호를 추천해주는 챗봇입니다.
검색된 데이터를 기반으로 패턴 분석 및 통계적 인사이트를 도출하여 예상 당첨번호를 추천하세요.

반드시 Context에 있는 패턴과 통계를 참고하여 분석하고, 랜덤 추천이 아닌 데이터 기반으로 답변하세요.
분석 시 다음 요소를 고려하세요:
- 최근 2년간 가장 많이 출현한 번호
- 연속 번호 출현 여부 및 출현 확률
- 짝수/홀수 비율 및 자주 등장하는 조합 패턴
- 과거 회차 중 현재 추천하는 번호와 유사한 패턴 분석

로또 관련 질문이 아니면 "로또 관련 질문을 해주세요."라고 답변하세요.

번호를 추천할 때는 반드시 아래 형식을 따르세요:
'추천번호는 [번호1, 번호2, 번호3, 번호4, 번호5, 번호6, 보너스번호]입니다.'

추천 이유도 함께 제공하세요. 추천 이유는 Context의 통계와 패턴을 기반으로 설명하세요.

Question: {question}
Context: {context}
Answer: """
)

prompt1 = ChatPromptTemplate.from_template(
    """
너는 로또 데이터 전문가야.
로또 관련질문이 오면 검색된 데이터를 기반으로 답변해야해.
답변할땐 어떤 데이터가 근거로 사용되었는지 같이 설명해야해.

Question: {question}
Context: {context}
Answer: """
)

class ChatbotAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        chat_history = Chatbot.objects.filter(author=request.user).order_by("-created_at")
        serializer = ChatbotSerializer(chat_history, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        serializer = ChatbotSerializer(data=request.data)
        if serializer.is_valid():
            user_input = serializer.validated_data["user_input"]
            chat_history = Chatbot.objects.filter(author=request.user).order_by("-created_at")
            
            message = [{"role": "system", "content": prompt}]
            
            for chat in chat_history:
                message.append({"role": "user", "content": chat.user_input})
                message.append({"role": "assistant", "content": chat.ai_response})
            
            message.append({"role": "user", "content": user_input})
            
            try:
                # 벡터 DB에서 관련 문서 검색
                docs = retriever.invoke(user_input)
                context = "\n".join([doc.page_content for doc in docs])
                
                # LLM 설정
                llm = ChatOpenAI(model="gpt-4o-mini")       
                
                # RAG 실행
                chain = prompt1 | llm
                ai_response = chain.invoke({"context": context, "question": user_input}).content
                
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            
            chatbot_instance = serializer.save(author=request.user, ai_response=ai_response)
            
            return Response(ChatbotSerializer(chatbot_instance).data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


