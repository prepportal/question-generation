import spacy

def extract_keywords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # Extract nouns and proper nouns as keywords
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return keywords

def find_title(text):
    keywords = extract_keywords(text)
    # Count occurrences of each keyword
    keyword_counts = {}
    for keyword in keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    # Find the most frequent keyword
    if keyword_counts:
        title = max(keyword_counts, key=keyword_counts.get)
        return title
    else:
        return "Title not found"