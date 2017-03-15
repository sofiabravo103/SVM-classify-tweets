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
            self.api.VerifyCredentials()

    @staticmethod
    def rate_limit_exceeded(error):
        for e in error.args[0]:
            if e['code'] == 88:
                return True
        return False

    @staticmethod
    def giving_up_error(error):
        for e in error.args[0]:
            if e['code'] in [144, 179]:
                return "{} ({})".format(e['message'], e['code'])
        return None

    def extract_and_clean_single(self, tweet_id, display_progress=None):
        """
        Extract text from the Twitter API and return the preprocessed text
        """
        if display_progress:
            progress = '[{}%]: '.format(round(display_progress, 2))
        else:
            progress = ''
        base_msg = '\r{}Extracting text from twitter api... '.format(progress)
        successfull = False
        print(base_msg, end="")
        while not successfull:
            try:
                json_tweet = self.api.GetStatus(tweet_id).AsDict()
            except twitter.error.TwitterError as e:
                giving_up_msg = TwitterPreprocessingInSpanish.giving_up_error(e)
                if giving_up_msg:
                    print('{}{}'.format(base_msg, giving_up_msg))
                    return None
                elif TwitterPreprocessingInSpanish.rate_limit_exceeded(e):
                    successfull = False
                    print('{}Rate limit exceeded, sleeping'.format(base_msg))
                    time.sleep(960)

                print('{}Retrying'.format(base_msg))
            else:
                successfull = True

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
