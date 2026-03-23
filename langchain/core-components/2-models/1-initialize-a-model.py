from langchain_openai import ChatOpenAI
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."
model = init_chat_model("gpt-5.2")
response = model.invoke("Why do parrots talk?")

# -------------------------------------------

os.environ["OPENAI_API_KEY"] = "sk-..."
model = ChatOpenAI(model="gpt-5.2")
response = model.invoke("Why do parrots talk?")
