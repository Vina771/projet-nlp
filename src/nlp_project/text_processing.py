import re

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:
    stopwords = None
    WordNetLemmatizer = None
    word_tokenize = None


DOMAIN_WORDS = {
    "not", "no", "never", "against", "war", "peace",
    "trump", "biden", "ukraine", "russia", "nato",
    "election", "vote", "democrat", "republican",
}

FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with",
}


def _stop_words():
    if stopwords is None:
        return FALLBACK_STOP_WORDS - DOMAIN_WORDS
    try:
        return set(stopwords.words("english")) - DOMAIN_WORDS
    except LookupError:
        return FALLBACK_STOP_WORDS - DOMAIN_WORDS


def _lemmatizer():
    return WordNetLemmatizer() if WordNetLemmatizer is not None else None


def clean_and_tokenize(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"^RT\s+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    try:
        tokens = word_tokenize(text) if word_tokenize else text.split()
    except LookupError:
        tokens = text.split()

    lemmatizer = _lemmatizer()
    stop_words = _stop_words()
    clean_tokens = []
    for token in tokens:
        if token in stop_words or len(token) <= 2:
            continue
        if lemmatizer is not None:
            try:
                token = lemmatizer.lemmatize(token)
            except LookupError:
                pass
        clean_tokens.append(token)

    return " ".join(clean_tokens)
