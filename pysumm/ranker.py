from abc import ABCMeta, abstractmethod
from nlputils import compute_cos_similarity
import numpy as np
from numpy import linalg as LA
import logging

class Ranker():
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def rank(self,doc_cluster,query=None):
        pass


class LexRank(Ranker):
    '''
    A query sensitive LexRank. 
    Refer to Biased LexRank: Passage retrieval using random walks with question-based priors[J]. Information Processing & Management, 2009, 45(1): 42-54.
    '''
    def __init__(self,relation_weight=0.85):
        '''
        Constructor
        '''
        self.relation_weight=relation_weight
        
    def rank(self,doc_cluster,query=None):
        doc_sentences=doc_cluster.total_sentences
        query_sentences=None
        if query:
            query_sentences=query.sents
        ranked_sent_list = self._rank_documents(doc_sentences,query_sentences)

        #the bigger the distance the better
        sorted_sents = sorted(ranked_sent_list, key=lambda sent: -sent.saliency)
        return sorted_sents
        
    def _rank_documents(self, sent_list,query_sentences):
        n = len(sent_list)
        #Initialises the adjacency matrix
        adjacency_matrix = np.zeros((n, n))
        
        degree = np.zeros(n)        
#         total_sentences=list(sent_list)
#         if query_sentences:
#             total_sentences.extend(query_sentences)       
        for i, senti in enumerate(sent_list):
            for j, sentj in enumerate(sent_list):
                if i == j:
                    adjacency_matrix[i][j] = 1                    
                elif i < j:
                    #adjacency_matrix[i][j] = self.sim_measurement.get_cos_similarity(senti.content, sentj.content)
                    adjacency_matrix[i][j]=compute_cos_similarity(senti.tf, sentj.tf)                  
                else:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i]

                degree[i] += adjacency_matrix[i][j]
                #if adjacency_matrix[i][j] > threshold:
                #    adjacency_matrix[i][j] = 1.0
                #    degree[i] += 1
                #else:
                #    adjacency_matrix[i][j] = 0        
        for i in xrange(n):
            for j in xrange(n):
                adjacency_matrix[i][j] = adjacency_matrix[i][j] / degree[i]
        inital_scores = np.zeros(n)
        if query_sentences:
            for i,senti in enumerate(sent_list):
                for sentj in query_sentences:
                    sim=compute_cos_similarity(senti.tf, sentj.tf)
                    inital_scores[i]+=sim               
            query_weight=inital_scores.sum()
            if query_weight!=0:
                inital_scores=inital_scores/query_weight
            else:
                inital_scores=np.ones(n,dtype=np.float64)/n
        else:
            inital_scores=np.ones(n,dtype=np.float64)/n
        scores = self.power_method(adjacency_matrix, inital_scores)
#         score_sum=scores.sum() 
#         if score_sum>1.001 or score_sum<0.999:
#             print score_sum       
        for i in xrange( n ):
            sent_list[i].saliency = scores[i]
        return sent_list
        
    def power_method(self, m,inital_scores ):               
        epsilon=0.001
        max_iter=10
        p=inital_scores.copy()
        inital_scores*=(1-self.relation_weight)
        iter_number = 0
        while True:
            iter_number += 1
            new_p = inital_scores+m.T.dot(p)*self.relation_weight       
#             for i in xrange( n ):
#                 for j in xrange( n ):
#                     new_p[i] += m[j][i] * p[j]*self.relation_weight            
            diff=LA.norm(new_p-p)
            p = new_p            
            logging.debug('lexrank error: ' + str(diff))       
            if diff < epsilon:
                break
            if iter_number>=max_iter: break
        logging.debug('lexrank converged after ' + str(iter_number) + ' iterations.')
        return p 

    
class ManifoldRank(Ranker):
    '''
    Manifold ranking.
    Refer to Manifold-Ranking Based Topic-Focused Multi-Document Summarization[C]//IJCAI. 2007, 7: 2903-2908.    
    We fix the inter-document weight to 1.
    Different from the original paper, we add an extra parameter query_weight to adjust the weight of query sentences.
    '''
    def __init__(self,relation_weight=0.85,intra_weight=0.8,query_weight=1.0):
        if query_weight<0 or query_weight>1:
            raise Exception('query weight must lie in [0,1]')        
        self.relation_weight=relation_weight
        self.intra_weight=intra_weight
        self.query_weight=query_weight
        return
    
    def rank(self,doc_cluster,query=None):
        doc_sentences=[]
        for index,doc in enumerate(doc_cluster.docs):
            for sent in doc.sents:
                sent.doc_id=index
                doc_sentences.append(sent)
        query_sentences=[]
        if query:
            query_sentences=query.sents
        for sent in query_sentences:
            sent.doc_id=-1            
        ranked_sent_list = self._rank_documents(doc_sentences,query_sentences)        

        #no_citing=[sent for sent in ranked_sent_list if sent.section_id>=0]
        sorted_sents = sorted(ranked_sent_list, key=lambda sent: -sent.saliency)
        return sorted_sents
        
    def _rank_documents(self, sent_list,query_sentences):                      
        total_sentences=sent_list+query_sentences
        n = len(total_sentences)
        adjacency_matrix = np.empty((n, n),dtype=np.float64)
        degree = np.zeros(n)       
        for i, senti in enumerate(total_sentences):
            for j, sentj in enumerate(total_sentences):
                if i == j:
                    adjacency_matrix[i][j] = 1                    
                elif i < j:
                    #adjacency_matrix[i][j] = self.sim_measurement.get_cos_similarity(senti.content, sentj.content)
                    similarity=compute_cos_similarity(senti.tf, sentj.tf)
                    if senti.doc_id!=sentj.doc_id:
                        similarity*=self.intra_weight
                    adjacency_matrix[i][j]=similarity                  
                else:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i]

                degree[i] += adjacency_matrix[i][j]
                #if adjacency_matrix[i][j] > threshold:
                #    adjacency_matrix[i][j] = 1.0
                #    degree[i] += 1
                #else:
                #    adjacency_matrix[i][j] = 0        
        for i in xrange(n):
            for j in xrange(n):
                # assert(degree[i] >= 1.)
                adjacency_matrix[i][j] /= degree[i]
                    
        initial_scores=np.empty(n)
        query_sent_weight=0
        if len(query_sentences)>0:
            query_sent_weight=self.query_weight/float(len(query_sentences))
        doc_sent_weight=(1.0-query_sent_weight*len(query_sentences))/len(sent_list)
        for i in xrange(n):
            if total_sentences[i].doc_id<0:
                initial_scores[i]=query_sent_weight
            else:
                initial_scores[i]=doc_sent_weight
        
        #initial_scores/=initial_scores.sum()
        initial_scores*=(1-self.relation_weight)
        scores = self.power_method(adjacency_matrix,initial_scores)
#         score_sum=scores.sum()
#         if score_sum>1.001 or score_sum<0.999:
#             print score_sum       
        for i in xrange( n ):
            total_sentences[i].saliency = scores[i]
        return sent_list
    
    def power_method(self, m,initial_scores):        
        epsilon=0.001
        max_iter=10
        p=initial_scores.copy()
        iter_number = 0
        while True:
            iter_number += 1
            new_p = initial_scores+m.T.dot(p)*self.relation_weight        
#             for i in xrange( n ):
#                 for j in xrange( n ):
#                     new_p[i] += m[j][i] * p[j]*self.relation_weight            
            diff=LA.norm(new_p-p)
            p = new_p            
            logging.debug('manifold error: ' + str(diff))       
            if diff < epsilon:
                break
            if iter_number>=max_iter: break
        logging.debug('manifold converged after ' + str(iter_number) + ' iterations.')
        return p   