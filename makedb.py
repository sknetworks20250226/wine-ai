from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone,ServerlessSpec
import os, re, pandas as pd
from typing import Dict


# 전처리 함수

def preprocess_text(text:str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 소문자 변환
    text = text.lower()
    
    # 특수문자 제거 (알파벳, 숫자, 공백은 유지)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 여러 개의 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# 메타데이터  생성(와인데이터와 함께)
from langchain_core.documents import Document
def create_document_with_metadata(row:pd.Series)->Document:
    # 제목과 설명을 결합
    title = str(row['title']) if pd.notna(row['title']) else ""
    description = str(row['description']) if pd.notna(row['description']) else ""
    text = f"{title} {description}"
    
    # 텍스트 전처리
    processed_text = preprocess_text(text)
    
    # 메타데이터 생성
    metadata = {
        'price': float(row['price']) if pd.notna(row['price']) else 0.0,
        'country': str(row['country']) if pd.notna(row['country']) else "Unknown",
        'title': title,  # 원본 제목 보존
        'description': description,  # 원본 설명 보존
        'points': float(row['points']) if pd.notna(row['points']) else 0.0,  # 와인 평점
        'variety': str(row['variety']) if pd.notna(row['variety']) else "Unknown",  # 와인 품종
        'winery': str(row['winery']) if pd.notna(row['winery']) else "Unknown",  # 와이너리
        'province': str(row['province']) if pd.notna(row['province']) else "Unknown",  # 지역
        'region_1': str(row['region_1']) if pd.notna(row['region_1']) else "Unknown",  # 세부 지역 1
        'region_2': str(row['region_2']) if pd.notna(row['region_2']) else "Unknown"   # 세부 지역 2
    }

    # 검색에 사용할 텍스트 생성 (더 많은 컨텍스트 포함)
    search_text = f"{title} {description} {metadata['variety']} {metadata['winery']} {metadata['province']} {metadata['region_1']} {metadata['region_2']}"
    processed_text = preprocess_text(search_text)

    return Document(
        page_content=processed_text,
        metadata=metadata
    )
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
# 인덱스가 생성되었다면 아래 코드를 주석으로 그렇지 않고 최초 실행이면 주석해제해서 인덱스 생성성
pc.create_index(
    name = os.environ['PINECONE_INDEX_NAME'],
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(
        region='us-east-1',
        cloud='aws'
    )
)

wine_index = pc.Index(os.environ['PINECONE_INDEX_NAME'])
print(wine_index.describe_index_stats())

# csv 파일 읽어서 벡터 db 만들기
# from langchain_community.document_loaders import CSVLoader
# loader = CSVLoader('winemag-data-130k-v2.csv', encoding='utf-8')
# docs = loader.load()
df = pd.read_csv('winemag-data-130k-v2.csv', encoding='utf-8')
df = df.dropna()
docs =  [create_document_with_metadata(row) for _, row in df.iterrows()]

# 임베딩 모델
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])

# 벡터화 해서 DB에 저장
from langchain_pinecone import PineconeVectorStore
import time

def upsert_batch(batch, index_name, embedding):
    try:
        # 기존 벡터 스토어 가져오기
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embedding,
            pinecone_api_key=os.environ['PINECONE_API_KEY']
        )
        
        # 배치 업서트
        vector_store.add_documents(batch)
        return True
    except Exception as e:
        print(f"Error in batch upsert: {str(e)}")
        return False

# 배치 크기와 재시도 설정
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# 진행 상황 추적
total_docs = len(docs)
processed_docs = 0
failed_batches = []

print(f"Starting to process {total_docs} documents...")

for i in range(0, total_docs, BATCH_SIZE):
    batch = docs[i:i+BATCH_SIZE]
    success = False
    retries = 0
    
    while not success and retries < MAX_RETRIES:
        try:
            success = upsert_batch(batch, os.environ['PINECONE_INDEX_NAME'], embedding)
            if success:
                processed_docs += len(batch)
                print(f"Progress: {processed_docs}/{total_docs} documents processed ({(processed_docs/total_docs)*100:.2f}%)")
            else:
                retries += 1
                if retries < MAX_RETRIES:
                    print(f"Retrying batch {i} to {i+len(batch)-1} (attempt {retries+1}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Error processing batch {i} to {i+len(batch)-1}: {str(e)}")
            retries += 1
            if retries < MAX_RETRIES:
                print(f"Retrying batch {i} to {i+len(batch)-1} (attempt {retries+1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
    
    if not success:
        failed_batches.append((i, i+len(batch)-1))
        print(f"Failed to process batch {i} to {i+len(batch)-1} after {MAX_RETRIES} attempts")

# 최종 결과 출력
print("\nProcessing completed!")
print(f"Successfully processed: {processed_docs}/{total_docs} documents")
if failed_batches:
    print("\nFailed batches:")
    for start, end in failed_batches:
        print(f"Batch {start} to {end}")

# 인덱스 통계 출력
wine_index = pc.Index(os.environ['PINECONE_INDEX_NAME'])
print("\nFinal index statistics:")
print(wine_index.describe_index_stats())
        