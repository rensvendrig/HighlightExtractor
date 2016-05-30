'''
Created on Apr 21, 2016

@author: Ziqiang
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import math

class TextSimilarity(object):
    '''
    classdocs
    '''

    def __init__(self,max_ngram=2,needStem=False):
        '''
        Constructor
        '''
        self.stemmer=PorterStemmer()
        if not needStem:
            self.vectorizer=CountVectorizer(stop_words = 'english',ngram_range=(1,max_ngram))
        else:
            self.vectorizer=CountVectorizer(analyzer=self.AnalyseText,ngram_range=(1,max_ngram))
        self.stop = stopwords.words('english')
        
    
    def get_cos_similarity(self,text1,text2):
        tfidf=self.vectorizer.fit_transform([text1,text2])
        cos_sim=cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return cos_sim    
    
    def calculate_TF(self,sents):
        all_finished=True
        for sent in sents:
            if not hasattr(sent, 'tf'):
                all_finished=False
                break
        if all_finished: return 
        texts=[sent.content for sent in sents]
        vectors=self.get_count_vector(texts)
        for i in xrange(len(sents)):
            sents[i].tf=vectors[i]
    
    def calculate_sentence_similarity(self,sent1,sent2):
        return self.get_similarity_from_vectors(sent1.tf, sent2.tf)
    
    def get_count_vector(self,texts):
        vectors=self.vectorizer.fit_transform(texts)
        return vectors
    
    def get_similarity_from_vectors(self,vector1,vector2):
        sim=cosine_similarity(vector1, vector2)[0][0]
        return sim
        
    def AnalyseText(self,doc):
        doc=doc.lower()
        doc=re.sub(r'[^a-z\d\s]',' ',doc)
        doc=re.sub(r'\d','#',doc)
        tokens=doc.split()
        stems=[]
        for t in tokens:
            if len(t)<2 or t in self.stop: continue
            stems.append(self.stemmer.stem(t))
        return stems

def compute_cos_similarity(dict1,dict2):
    sum_common=0.
    sum1=0
    for k,v1 in dict1.items():
        if not dict2.has_key(k): continue
        v2=dict2[k]
        sum_common+=v1*v2
        sum1+=v1*v1
    if sum_common==0: return 0
    sum2=sum([v*v for v in dict2.values()])
    m=math.sqrt(sum1*sum2)
    return sum_common/m
    
if __name__ == '__main__':
    dict1={'a':0.2,'b':0.7,'c':1.3}
    dict2={'a':0.9,'b':0.1,'d':1.3}
    sim=compute_cos_similarity(dict1, dict2)
    print sim