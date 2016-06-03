import ranker
from nlputils.factory import build_common_document_converter
from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Normalizer

class SVRRanker(ranker.Ranker):
    def rank(self, doc_cluster, query=None):
        if query:
            raise Exception('SVR embeds query in the attribute doc_cluster.query')
        return
    
    def __init__(self):
        return
    
class Feature():
    @abstractmethod
    def Extract(self,instance):
        pass



class SentenceFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,features):
        self.features=features        
    def fit(self, x, y=None):
        return self
    def transform(self,instances):
        return

class InstanceCreator(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self,clusters):
        '''
        return [instance list]
        each list stands for a document cluster
        '''
        instance_list=[]
        for cluster in clusters:
            cluster_instances=[]
            instance_list.append(cluster_instances)
            for doc in cluster.docs:
                for sent in doc.sents:
                    instance=RankingInstance(cluster,doc,sent)
                    cluster_instances.append(instance)
        return instance_list
        
class RankingInstance(object):
    def __init__(self,cluster,document,sentence):
        self.cluster=cluster
        self.document=document
        self.sentence=sentence
        return

