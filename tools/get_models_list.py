from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#Нет 4o!!!!!

models = client.models.list()

for model in models:
    print(model.id)

