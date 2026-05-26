import re
import html
from stop_words import get_stop_words


class TextPreprocessor:
    def __init__(self, language: str = 'ru'):
        self.stopwords = set(get_stop_words(language))
        self._url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self._email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self._phone_pattern = re.compile(r'\+?\d[\d\-\(\)\s]{7,}')

    def clean_text(self, text: str,
                   remove_stopwords: bool = True,
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phones: bool = True,
                   normalize_yo: bool = True) -> str:

        if not text or not isinstance(text, str):
            return ""

        text = html.unescape(text)
        text = re.sub(r'<.*?>', ' ', text)

        if remove_urls:
            text = self._url_pattern.sub(' ', text)
        if remove_emails:
            text = self._email_pattern.sub(' ', text)
        if remove_phones:
            text = self._phone_pattern.sub(' ', text)

        text = text.lower()
        if normalize_yo:
            text = text.replace('ё', 'е')

        text = re.sub(r'[^а-яa-z0-9\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if remove_stopwords:
            words = [w for w in text.split() if w not in self.stopwords and len(w) > 1]
            text = ' '.join(words)

        return text


preprocessor = TextPreprocessor()
