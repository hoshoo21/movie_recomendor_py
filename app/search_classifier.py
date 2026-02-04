import re 


def classify_search_intent (query:str):
    query = query.lower().strip()
    has_year = re.search(r'\b(19|20)\d{2}\b', query)
    
    vibe_words = {'about', 'like', 'feel', 'vibe', 'similar', 'story', 'meaning'}
    query_words = set(query.split())
    has_vibe = not query_words.isdisjoint(vibe_words)
    
    if len(query_words) <= 2 and not has_vibe:
        return {"strategy": "LEXICAL_HEAVY", "lexical_weight": 0.8, "semantic_weight": 0.2}
    
    if has_vibe or len(query_words) > 5:
        return {"strategy": "SEMANTIC_HEAVY", "lexical_weight": 0.2, "semantic_weight": 0.8}
    
    return {"strategy": "HYBRID", "lexical_weight": 0.5, "semantic_weight": 0.5}