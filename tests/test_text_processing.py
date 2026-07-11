from src.nlp_project.text_processing import clean_and_tokenize


def test_clean_and_tokenize_removes_twitter_noise():
    text = "RT @user The election is NOT over! #Vote https://example.com"

    tokens = clean_and_tokenize(text)

    assert "user" not in tokens
    assert "https" not in tokens
    assert "election" in tokens
    assert "not" in tokens
    assert "vote" in tokens
