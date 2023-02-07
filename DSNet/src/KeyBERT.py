import numpy as np
from transformers import BertModel
from keybert import KeyBERT

# KeyBERT 알고리즘
class Rank(object):
    def get_keywords(self, sentences):
        sentence = ' '.join(sentences)
        
        model = BertModel.from_pretrained('skt/kobert-base-v1')
        # model = 'distiluse-base-multilingual-cased-v1'
        kw_extractor = KeyBERT(model)

        keywords = kw_extractor.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1), 
            stop_words=None, 
            top_n=50)
        
        return keywords

# KeyBERT hashtag Class
class KeyBERTModel(object):
    def __init__(self, text):
                
        self.rank = Rank()
        self.keywords = self.rank.get_keywords(text)
    
    def get_hashtags(self, word_num=10):
        hashtags = []
        
        for keyword, score in self.keywords[:word_num]:
            hashtags.append(keyword)
        
        #hashtags[0] = hashtags[0]+'브이로그'
        
        return hashtags