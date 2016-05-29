from abc import ABCMeta, abstractmethod
from nlputils.data_structure import Document
import re

LIMIT_SENTENCE=1
LIMIT_TOKEN=2

class Selector(object):

    def __init__(self,sent_filter,measurement):
        self.measurement=measurement
        self.sent_filter=sent_filter
        return
    
    def select(self,ranked_sentences,limit_mode,limit_count):
        '''
        limit_mode can be 1(sentences) or 2(tokens)
        We do not truncate the last sentence. Therefore it may exceed the limit_count in the token mode.
        '''
        summary_sents=[]
        count=0
        for sent in ranked_sentences:
            sent.rouge_tokens=rouge_tokenize(sent)
            sent=self.sent_filter.filter_sentence(sent)
            if not sent: continue
            if self.measurement.is_redudant(sent): continue
            self.measurement.put_into_summary(sent)
            if limit_mode==LIMIT_SENTENCE:
                count+=1
            elif limit_mode==LIMIT_TOKEN:
                count+=len(sent.rouge_tokens)
            else:
                raise Exception('Invaid mode:{}'.format(limit_mode))
            summary_sents.append(sent)
            if count>=limit_count: break
        summary=Document(sents=summary_sents)
        return summary

class RedudancyMeasurement():
    __metaclass__ = ABCMeta
    @abstractmethod
    def is_redudant(self,sent):
        pass
    @abstractmethod
    def put_into_summary(self,sent):
        pass
    
    def check_put(self,sent):
        if self.is_redudant(sent): return False
        self.put_into_summary(sent)
        return True     
    
class NgramMeasurement(RedudancyMeasurement):
    def __init__(self,gram_size=2,threshold=0.5):
        self.gram_size=gram_size
        self.threshold=threshold
        self.current_ngrams=set()
        return
    
    def to_ngrams(self,sent):
        tokens=sent.rouge_tokens
        ngrams=[]
        for i in xrange(len(tokens)-self.gram_size+1):
            ngram=tokens[i:i+self.gram_size]
            ngrams.append(' '.join(ngram))
        return ngrams
        
    def is_redudant(self,sent):
        ngrams=self.to_ngrams(sent)
        gram_count=len(ngrams)
        exist_count=len([ngram for ngram in ngrams if ngram in self.current_ngrams])
        return exist_count>=gram_count*self.threshold
    
    def put_into_summary(self,sent):
        ngrams=self.to_ngrams(sent)
        for ngram in ngrams:
            self.current_ngrams.add(ngram)
    
class SummarySentenceFilter(object):
    def __init__(self,min_length=9):
        self.min_length=min_length
        return
    def filter_sentence(self,sent):
        if len(sent.rouge_tokens)<self.min_length: return None
        return sent
    
def rouge_tokenize(sent):
    content=sent.content.lower()
    content=content.replace('-',' ')
    content=re.sub(r'[^a-z0-9\s]','',content)
    tokens=content.split()
    return tokens