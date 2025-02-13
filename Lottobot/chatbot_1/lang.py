import os
import getpass
from langchain_openai import ChatOpenAI
from django.conf import settings
import openai


os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


# model 생성
model = ChatOpenAI(model="gpt-4o-mini")

# prompt
prompt = "말티즈의 고향은 어디야?"

# chain 실행
answer = model.invoke(prompt)
print(answer)
