from abc import ABCMeta, abstractmethod
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import data_structure

class SentenceProcesser():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def process(self,sentence):
        pass

class Tokenizer(SentenceProcesser):
    def process(self, sentence):
        tokens=word_tokenize(sentence.content)
        words=[]
        for token in tokens:
            word=data_structure.Word(token)
            words.append(word)
        sentence.words=words
        return sentence

class Stemmer(SentenceProcesser):
    def __init__(self):
        self.stemmer=PorterStemmer()
    def process(self, sentence):
        for word in sentence.words:
            word.stem=self.stemmer.stem(word.content)
        return sentence
    
class  TFCounter(SentenceProcesser):
    def __init__(self,stopwords=None,use_stem=False,ngram_range=(1,)):
        self.stopwords=stopwords
        self.use_stem=use_stem
        self.ngram_range=ngram_range
        return
    def process(self, sentence):
        tf_map=defaultdict(int)
        term_list=[]
        for word in sentence.words:
            text=word.content.lower()
            if self.use_stem:
                text=word.stem
            term_list.append(text)
        for n in self.ngram_range:
            ngrams=to_ngram(term_list, n, self.stopwords)
            for ngram in ngrams:
                tf_map[ngram]+=1
        sentence.tf=tf_map
        return sentence

def to_ngram(token_list,n,stopwords=None):
    length=len(token_list)
    ngrams=[]
    for i in xrange(length):
        if i+n>length: break
        if not stopwords:
            ngrams.append(' '.join(token_list[i:i+n]))
            continue
        has_stop=False
        for j in xrange(n):
            token=token_list[i+j]
            if token in stopwords:
                has_stop=True
                break
        if has_stop: continue
        ngrams.append(' '.join(token_list[i:i+n]))
    return ngrams           

class SentContentFormatter():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def format(self,text):
        pass

class SplitFormatter(SentContentFormatter):
    def format(self, text):
        text=' '.join(text.split())
        return text

class SentenceConverter(object):
    def __init__(self,content_formatters=[],sent_processors=[],min_length=1):
        self.content_formatters=content_formatters
        self.sent_processors=sent_processors
        self.min_length=min_length
        return
    
    def convert_to_sentence(self,content):
        for formatter in self.content_formatters:
            content=formatter.format(content)
            if len(content)<self.min_length: return None
        sent=data_structure.Sentence(content)
        for processor in self.sent_processors:
            sent=processor.process(sent)
        return sent
    
        
if __name__=='__main__':
    mytokenizer=Tokenizer()
    content='This is haha located in apples.'
    sent=data_structure.Sentence(content)
    mytokenizer.process(sent)
    print ' '.join([word.content for word in sent.words])   