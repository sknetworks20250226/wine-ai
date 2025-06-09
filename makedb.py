from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone,ServerlessSpec
import os, re, pandas as pd
from typing import Dict


# 전처리 함수

def preprocess_text(text:str) -> str:
    text = text.lower()
    # 특수문자 제거
    text = re.sub(r'[\w\s]','',text)
    # 여러개 공백을 하나로
    text = re.sub(r'\s+',' ',text)
    return text.strip()
# 메타데이터  생성(와인데이터와 함께)
def create_document_with_metadata(row:pd.Series)->Dict:
    text = f"{row['title']} {row['description']}"
    processed_text = preprocess_text(text)
    metadata = {
        'price' : None,
        'country' : None  # 메타정보로 더 사용할 게 있으면 추가
    }

    return {
        'page_content':processed_text,
        'metadata':metadata
    }
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# pc.create_index(
#     name = os.environ['PINECONE_INDEX_NAME'],
#     dimension=1536,
#     metric='cosine',
#     spec=ServerlessSpec(
#         region='us-east-1',
#         cloud='aws'
#     )
# )

wine_index = pc.Index(os.environ['PINECONE_INDEX_NAME'])
print(wine_index.describe_index_stats())

# csv 파일 읽어서 벡터 db 만들기
# from langchain_community.document_loaders import CSVLoader
# loader = CSVLoader('winemag-data-130k-v2.csv', encoding='utf-8')
# docs = loader.load()
df = pd.read_csv('winemag-data-130k-v2.csv', encoding='utf-8')
docs =  [create_document_with_metadata(row) for _, row in df.iterrows()]

# 임베딩 모델
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])

# 벡터화 해서 DB에 저장
from langchain_pinecone import PineconeVectorStore
BATCH_SIZE = 100
for i in range(0,len(docs), BATCH_SIZE):
    batch = docs[i : i+BATCH_SIZE]
    # try:
    PineconeVectorStore.from_documents(
        documents=batch,
        index_name = os.environ['PINECONE_INDEX_NAME'],
        embedding=embedding
    )
    print(f'{i} ~ {i+len(batch)-1} documents indexed')
    # except Exception as e:
    #     print(f'{i} ~ {i+len(batch)-1}  error : {e}')
        