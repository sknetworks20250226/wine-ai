from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
##############################################################
load_dotenv()
llm = ChatOpenAI(model = os.environ['OPENAI_LLM_MODEL'],temperature=0.2)
embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])
vector_store = PineconeVectorStore(
    index_name= os.environ['PINECONE_INDEX_NAME'],
    embedding=embeddings,
    pinecone_api_key = os.environ['PINECONE_API_KEY']
)

# 이미지에서 정보추출
def describe_dish_flavor(image_bytes,query):
    data_url = f'data:image/jpeg;base64,{image_bytes}'
    messages = [
        {'role':'system','content':'''
            You are an expert sommelier and food critic specializing in wine pairing. Your task is to analyze dishes and identify their key characteristics that are crucial for wine pairing.

            Focus on these specific aspects:
            1. Primary Taste Profile:
               - Sweetness level (low/medium/high)
               - Acidity level (low/medium/high)
               - Saltiness level (low/medium/high)
               - Umami presence (yes/no, intensity)
               - Spice level (low/medium/high)

            2. Texture and Mouthfeel:
               - Main texture (creamy/crispy/tender/etc.)
               - Richness level (light/medium/heavy)
               - Temperature (hot/cold/room temperature)

            3. Key Ingredients and Their Impact:
               - List main ingredients
               - Note any dominant flavors
               - Identify any unique or strong flavor elements

            4. Cooking Method and Preparation:
               - Main cooking technique
               - Any special preparation methods
               - Sauce or seasoning presence

            Format your response in a structured way:
            1. Start with a brief overview of the dish
            2. List the key characteristics in the above categories
            3. End with a summary of the most important elements for wine pairing

            Remember: Focus on characteristics that are most relevant for wine pairing decisions.
         '''},
         {'role':'user','content':[
             {'type':'text','text':query},             
             {'type':'image_url','image_url':{'url':data_url}}
         ]}
    ]
    return llm.invoke(messages).content

# 벡터DB에서 검색
def search_wine(dish_flavor):
    # 와인 페어링 전문 지식 기반 키워드 시스템
    wine_pairing_keywords = {
        'taste': {
            'sweet': ['sweet', 'sugar', 'honey', 'caramel', 'fruity', 'ripe', 'sweetness'],
            'sour': ['sour', 'acidic', 'citrus', 'tart', 'tangy', 'fresh', 'bright'],
            'spicy': ['spicy', 'hot', 'pepper', 'chili', 'pungent', 'heat', 'spice'],
            'rich': ['rich', 'heavy', 'bold', 'intense', 'full-bodied', 'deep', 'concentrated'],
            'light': ['light', 'delicate', 'subtle', 'mild', 'refreshing', 'crisp', 'elegant'],
            'umami': ['umami', 'savory', 'earthy', 'mushroom', 'meaty', 'savory', 'complex'],
            'bitter': ['bitter', 'dark chocolate', 'coffee', 'herbal', 'bitterness', 'tannic']
        },
        'texture': {
            'creamy': ['creamy', 'smooth', 'velvety', 'buttery', 'rich', 'soft'],
            'crispy': ['crispy', 'crunchy', 'fresh', 'crackling', 'crisp'],
            'tender': ['tender', 'soft', 'delicate', 'juicy', 'succulent']
        },
        'wine_type': {
            'white': ['white', 'chardonnay', 'sauvignon blanc', 'riesling', 'pinot grigio'],
            'red': ['red', 'cabernet', 'merlot', 'pinot noir', 'syrah'],
            'sparkling': ['sparkling', 'champagne', 'prosecco', 'cava'],
            'rose': ['rose', 'blush', 'pink']
        }
    }

    # 요리 설명 분석 및 특징 추출
    dish_characteristics = {}
    for category, subcategories in wine_pairing_keywords.items():
        for subcategory, keywords in subcategories.items():
            # 부분 일치도 고려
            matches = sum(1 for keyword in keywords if keyword in dish_flavor.lower())
            if matches > 0:
                dish_characteristics[subcategory] = matches * 2

    # 와인 페어링 규칙 기반 검색 쿼리 생성
    enhanced_query = f"Expert wine pairing for dish: {dish_flavor}\n"
    enhanced_query += "Key characteristics for wine pairing:\n"
    
    # 주요 특징 강조
    for subcategory, weight in dish_characteristics.items():
        enhanced_query += f"- {subcategory} (importance: {weight})\n"
    
    # 와인 페어링 규칙 추가
    enhanced_query += "\nWine pairing rules:\n"
    if any(k in dish_characteristics for k in ['spicy', 'hot']):
        enhanced_query += "- Look for wines with residual sugar to balance spice\n"
    if any(k in dish_characteristics for k in ['rich', 'heavy']):
        enhanced_query += "- Consider full-bodied wines to match intensity\n"
    if any(k in dish_characteristics for k in ['light', 'delicate']):
        enhanced_query += "- Prefer light, crisp wines to complement delicacy\n"
    if any(k in dish_characteristics for k in ['sour', 'acidic']):
        enhanced_query += "- Match with wines of similar acidity\n"

    # 메타데이터 필터링 조건 설정
    metadata_filters = {
        'price': {'$gte': 0},
        'country': {'$exists': True}
    }

    # 유사도 검색 실행
    results_with_scores = vector_store.similarity_search_with_score(
        enhanced_query,
        k=30,  # 더 많은 후보 검색
        filter=metadata_filters
    )

    # 결과 정렬 및 필터링
    wine_reviews = []
    for doc, score in results_with_scores:
        metadata = doc.metadata
        wine_content = doc.page_content.lower()
        
        # 키워드 매칭 점수 계산 (부분 일치 고려)
        keyword_score = 0
        total_possible_matches = 0
        for subcategory, weight in dish_characteristics.items():
            keywords = []
            for category in wine_pairing_keywords.values():
                if subcategory in category:
                    keywords.extend(category[subcategory])
            
            # 부분 일치 점수 계산
            matches = sum(1 for keyword in keywords if keyword in wine_content)
            if matches > 0:
                keyword_score += weight * matches
            total_possible_matches += weight * len(keywords)

        # 와인 타입 매칭 점수 추가
        wine_type_score = 0
        for wine_type, keywords in wine_pairing_keywords['wine_type'].items():
            if any(keyword in wine_content for keyword in keywords):
                wine_type_score += 1

        # 점수 정규화 (0-1 범위)
        normalized_keyword_score = keyword_score / total_possible_matches if total_possible_matches > 0 else 0
        normalized_wine_type_score = wine_type_score / len(wine_pairing_keywords['wine_type'])

        # 최종 점수 계산
        # 기본 유사도 점수에 매칭 점수를 가중치로 적용
        final_score = score * (1 + normalized_keyword_score * 0.5 + normalized_wine_type_score * 0.3)
        # 최종 점수를 0-1 범위로 제한
        final_score = min(1.0, final_score)

        review_text = f'''유사도: {final_score:.4f}
            와인명: {metadata.get('title', 'Unknown')}
            국가: {metadata.get('country', 'N/A')}
            가격: ${metadata.get('price', 'N/A')}
            맛 매칭 점수: {normalized_keyword_score:.4f}
            와인 타입 매칭 점수: {normalized_wine_type_score:.4f}
            내용: {metadata.get('description', 'No description available')}'''
        wine_reviews.append((final_score, review_text))

    # 최종 점수로 정렬하여 상위 2개 선택
    wine_reviews.sort(key=lambda x: x[0], reverse=True)
    return {
        'dish_flavor': dish_flavor,
        'wine_reviews': '\n\n'.join(review for _, review in wine_reviews[:2])
    }

# 추천사유를 llm으로 생성
def recommand_wine(inputs):
    prompt = ChatPromptTemplate.from_messages([
        ('system','''You are an expert sommelier specializing in wine and food pairing. Your task is to analyze the dish characteristics and wine reviews to provide a detailed wine pairing recommendation.

            Follow these steps in your analysis:
            1. Dish Analysis:
               - Identify the key flavor profiles
               - Note the texture and richness
               - Consider the cooking method
               - Highlight any challenging elements for wine pairing

            2. Wine Evaluation:
               - Analyze each wine's characteristics
               - Compare the wines' profiles
               - Consider the price point
               - Evaluate the regional pairing

            3. Pairing Justification:
               - Explain why each wine would work with the dish
               - Consider both complementary and contrasting elements
               - Address any potential challenges
               - Provide specific pairing notes

            4. Final Recommendation:
               - Select the best overall match
               - Provide detailed reasoning
               - Include serving suggestions if relevant

            Format your response in Korean, following this structure:
            1. 요리 분석
            2. 와인 평가
            3. 페어링 근거
            4. 최종 추천
            '''),
        ('human',''' 
            아래의 '요리의 풍미' 와 '와인 리뷰'를 바탕으로 전문적인 와인 페어링 추천을 해주세요.
            각 와인의 특성과 요리의 특성을 상세히 분석하고, 페어링이 적절한 이유를 설명해주세요.
            최종적으로 가장 적합한 와인을 추천하고, 그 이유를 명확히 설명해주세요.
         
            '요리의 풍미':
            {dish_flavor}
         
            '와인 리뷰':
            {wine_reviews}
            ''')
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)
