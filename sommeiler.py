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
            Persona:
            As a flavor analysis system, I am equipped with a deep understanding of food ingredients, cooking methods, and sensory properties such as taste, texture, and aroma. I can assess and break down the flavor profiles of dishes by identifying the dominant tastes (sweet, sour, salty, bitter, umami) as well as subtler elements like spice levels, richness, freshness, and aftertaste. I am able to compare different foods based on their ingredients and cooking techniques, while also considering cultural influences and typical pairings. My goal is to provide a detailed analysis of a dish's flavor profile to help users better understand what makes it unique or to aid in choosing complementary foods and drinks.

            Role:

            1. Flavor Identification: I analyze the dominant and secondary flavors of a dish, highlighting key taste elements such as sweetness, acidity, bitterness, saltiness, umami, and the presence of spices or herbs.
            2. Texture and Aroma Analysis: Beyond taste, I assess the mouthfeel and aroma of the dish, taking into account how texture (e.g., creamy, crunchy) and scents (e.g., smoky, floral) contribute to the overall experience.
            3. Ingredient Breakdown: I evaluate the role each ingredient plays in the dish's flavor, including their impact on the dish's balance, richness, or intensity.
            4. Culinary Influence: I consider the cultural or regional influences that shape the dish, understanding how traditional cooking methods or unique ingredients affect the overall taste.
            5. Food and Drink Pairing: Based on the dish's flavor profile, I suggest complementary food or drink pairings that enhance or balance the dish's qualities.

            Examples:

            - Dish Flavor Breakdown:
            For a butter garlic shrimp, I identify the richness from the butter, the pungent aroma of garlic, and the subtle sweetness of the shrimp. The dish balances richness with a touch of saltiness, and the soft, tender texture of the shrimp is complemented by the slight crispness from grilling.

            - Texture and Aroma Analysis:
            A creamy mushroom risotto has a smooth, velvety texture due to the creamy broth and butter. The earthy aroma from the mushrooms enhances the umami flavor, while a sprinkle of Parmesan adds a savory touch with a mild sharpness.

            - Ingredient Role Assessment:
            In a spicy Thai curry, the coconut milk provides a rich, creamy base, while the lemongrass and lime add freshness and citrus notes. The chilies bring the heat, and the balance between sweet, sour, and spicy elements creates a dynamic flavor profile.

            - Cultural Influence:
            A traditional Italian margherita pizza draws on the classic combination of fresh tomatoes, mozzarella, and basil. The simplicity of the ingredients allows the flavors to shine, with the tanginess of the tomato sauce balancing the richness of the cheese and the freshness of the basil.

            - Food Pairing Example:
            For a rich chocolate cake, I would recommend a sweet dessert wine like Port to complement the bitterness of the chocolate, or a light espresso to contrast the sweetness and enhance the richness of the dessert.
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
        'taste_profile': {
            'sweet': ['sweet', 'sugar', 'honey', 'caramel', 'fruity', 'sweetness', 'ripe'],
            'sour': ['sour', 'acidic', 'citrus', 'tart', 'tangy', 'freshness', 'bright'],
            'spicy': ['spicy', 'hot', 'pepper', 'chili', 'pungent', 'spice', 'heat'],
            'rich': ['rich', 'heavy', 'bold', 'intense', 'full-bodied', 'depth', 'concentrated'],
            'light': ['light', 'delicate', 'subtle', 'mild', 'refreshing', 'crisp', 'elegant'],
            'umami': ['umami', 'savory', 'earthy', 'mushroom', 'meaty', 'savory', 'complex'],
            'bitter': ['bitter', 'dark chocolate', 'coffee', 'herbal', 'bitterness', 'tannic']
        },
        'texture': {
            'creamy': ['creamy', 'smooth', 'velvety', 'buttery', 'rich'],
            'crispy': ['crispy', 'crunchy', 'crackling', 'fresh'],
            'tender': ['tender', 'soft', 'delicate', 'juicy'],
            'chewy': ['chewy', 'firm', 'dense', 'substantial']
        },
        'aroma': {
            'fruity': ['fruity', 'berry', 'citrus', 'tropical'],
            'floral': ['floral', 'perfumed', 'aromatic', 'fragrant'],
            'herbal': ['herbal', 'green', 'vegetal', 'fresh'],
            'spicy': ['spicy', 'peppery', 'warm', 'aromatic']
        },
        'cooking_method': {
            'grilled': ['grilled', 'charred', 'smoky', 'barbecued'],
            'fried': ['fried', 'crispy', 'golden', 'deep-fried'],
            'steamed': ['steamed', 'delicate', 'light', 'fresh'],
            'braised': ['braised', 'rich', 'tender', 'slow-cooked']
        }
    }

    # 요리 설명 분석 및 특징 추출
    dish_characteristics = {
        'taste_profile': {},
        'texture': {},
        'aroma': {},
        'cooking_method': {}
    }

    # 각 카테고리별 특징 분석 및 가중치 계산
    for category, subcategories in wine_pairing_keywords.items():
        for subcategory, keywords in subcategories.items():
            weight = sum(2 for keyword in keywords if keyword in dish_flavor.lower())
            if weight > 0:
                dish_characteristics[category][subcategory] = weight

    # 와인 페어링 전문가 관점의 검색 쿼리 생성
    enhanced_query = f"Expert wine pairing recommendation for a dish with the following characteristics:\n"
    
    # 주요 특징 강조
    for category, characteristics in dish_characteristics.items():
        if characteristics:
            enhanced_query += f"\n{category.replace('_', ' ').title()}:\n"
            for char, weight in characteristics.items():
                enhanced_query += f"- {char} ({weight}): {', '.join(wine_pairing_keywords[category][char][:3])}\n"

    # 와인 페어링 규칙 추가
    enhanced_query += "\nWine pairing rules:\n"
    enhanced_query += "1. Match intensity of wine with dish\n"
    enhanced_query += "2. Consider acidity balance\n"
    enhanced_query += "3. Complement or contrast flavors\n"
    enhanced_query += "4. Consider regional pairings\n"

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
        
        # 각 카테고리별 매칭 점수 계산
        category_scores = {}
        for category, subcategories in wine_pairing_keywords.items():
            category_score = 0
            for subcategory, keywords in subcategories.items():
                if subcategory in dish_characteristics[category]:
                    weight = dish_characteristics[category][subcategory]
                    keyword_matches = sum(1 for keyword in keywords if keyword in wine_content)
                    category_score += weight * keyword_matches
            category_scores[category] = category_score

        # 최종 점수 계산 (유사도 + 카테고리별 매칭 점수)
        final_score = score
        for category_score in category_scores.values():
            final_score += category_score * 0.2  # 각 카테고리 점수에 가중치 부여

        # 점수 정규화 (0-1 범위로)
        normalized_score = min(1.0, final_score / 2)  # 최대 점수를 2로 가정하고 정규화

        review_text = f'''유사도: {normalized_score:.4f}
            와인명: {metadata.get('title', 'Unknown')}
            국가: {metadata.get('country', 'N/A')}
            가격: ${metadata.get('price', 'N/A')}
            맛 매칭 점수: {sum(category_scores.values())}
            내용: {metadata.get('description', 'No description available')}'''
        wine_reviews.append((normalized_score, review_text))

    # 최종 점수로 정렬하여 상위 2개 선택
    wine_reviews.sort(key=lambda x: x[0], reverse=True)
    return {
        'dish_flavor': dish_flavor,
        'wine_reviews': '\n\n'.join(review for _, review in wine_reviews[:2])
    }

# 추천사유를 llm으로 생성
def recommand_wine(inputs):
    prompt = ChatPromptTemplate.from_messages([
        ('system',''' '''),
        ('human',''' 
            와인 페어링 추천에 아래의 '요리의 풍미' 와 '와인 리뷰' 를 참고해 한글로 답변해 주세요
            추천된 와인이 두개여야 하고 이를 검증한 다음에 검증이 되면
            두개의 추천된 와인중에 가장 어울리는 와인을 추천해 주세요
            위의 두 과정으 다시 한번 검토해서 의도대로 답변 해 주세요
         
            '요리의 풍미':
            {dish_flavor}
         
            '와인 리뷰':
            {wine_reviews}
            ''')
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)
