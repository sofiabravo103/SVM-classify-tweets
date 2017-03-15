import twitter
import time
from nltk.corpus import stopwords


class TwitterPreprocessingInSpanish():
    REJECTED_WORDS = ['rt', '(cont)']
    REJECTED_PATTERS = ['http', '@', '#']
    REJECTED_CHARS = [
        '!', '=', '|', '&', '*', '"', '>', '<', '(', ')',
        ':', '.', '/', '\\', ',', '[', ']', '-', ';'
    ]

    def __init__(self, api_info=None):
        if api_info:
            self.api = twitter.Api(
                consumer_key=api_info['consumer_key'],
                consumer_secret=api_info['consumer_secret'],
                access_token_key=api_info['access_token_key'],
                access_token_secret=api_info['access_token_secret']
            )
            print(self.api.VerifyCredentials())

    def extract_and_clean_single(self, tweet_id):
        """
        Extract text from the Twitter API and return the preprocessed text
        """
        successfull = False
        print('Extrating text from twitter api...')
        while not successfull:
            try:
                json_tweet = self.api.GetStatus(tweet_id).AsDict()
            except twitter.error.TwitterError as e:
                if e[0][0]['code'] == 88:
                    successfull = False
                    print('\rRate limit exceeded, I will sleep now.')
                    time.sleep(960)
                print('\rRetrying...')
            else:
                successfull = True
                print('\rDone.')

        return TwitterPreprocessingInSpanish.clean(json_tweet['text'])

    @staticmethod
    def is_number(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    @classmethod
    def clean(cls, text):
        """
        Returns the preprocessed text extratected from the text parameter.
        """
        bag = []
        for word in text.split():
            word = word.lower()
            ok_word = True
            for patt in cls.REJECTED_PATTERS:
                if patt in word:
                    ok_word = False
                    break

        if ok_word and word not in cls.REJECTED_WORDS:
            ready_word = ''
            for i in range(0, len(word)):
                if (word[i] not in cls.REJECTED_CHARS and not
                        TwitterPreprocessingInSpanish.is_number(word[i])):
                    ready_word += word[i]

            if ready_word not in stopwords.words('spanish'):
                bag.append(ready_word)

        return ' '.join(bag)
