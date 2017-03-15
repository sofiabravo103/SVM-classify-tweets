from sklearn.feature_extraction.text import CountVectorizer


class BagOfWords():
    def __init__(self, sentences, min_n=1, max_n=1):
        self.ngram = CountVectorizer(ngram_range=(min_n, max_n), token_pattern=r'\b\w+\b')

    def train_and_extract(self, sentences):
        return self.ngram.fit_transform(sentences)

    def extract(self, sentences):
        return self.ngram.transform(sentences)
