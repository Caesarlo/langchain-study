

import os

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

load_dotenv()


api_key = os.getenv("SILICONFLOW_API_KEY")
base_url = os.getenv("SILICONFLOW_BASE_URL")


model_1 = init_chat_model(model_provider="openai",
                          model="Pro/MiniMaxAI/MiniMax-M2.5", api_key=api_key, base_url=base_url)
model_2 = init_chat_model(model_provider="openai",
                          model="Pro/zai-org/GLM-5", api_key=api_key, base_url=base_url)

callback = UsageMetadataCallbackHandler()
result_1 = model_1.invoke("Hello", config={"callbacks": [callback]})
result_2 = model_2.invoke("Hello", config={"callbacks": [callback]})
print(callback.usage_metadata)
