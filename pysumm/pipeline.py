import multiprocessing as mul

class SummarizationPipeline(object):
    def __init__(self,ranker,selector):
        self.ranker=ranker
        self.selector=selector
        return
    
    def summarize(self,doc_cluster,query=None,length_mode=2,length_limit=100):
        summary=summarize(doc_cluster,query,self,length_mode,length_limit)
        return summary
    
    def batch_summarize(self,doc_cluster_list,query_list,length_mode=2,length_limit=100):
        pool=mul.Pool()
        result_list=[]
        for doc_cluster,query in zip(doc_cluster_list,query_list):
            result=pool.apply_async(summarize, args=(doc_cluster, query, self,length_mode, length_limit))
            result_list.append(result)
        pool.close()
        pool.join()
        summaries=[]
        for result in result_list:
            summaries.append(result.get())
        return summaries    
    
def summarize(doc_cluster,query,pipeline,length_mode,length_limit):
    ranked_sentences=pipeline.ranker.rank(doc_cluster, query)
        #logging.info('Sentence selection')
    result=pipeline.selector.select(ranked_sentences,length_mode, length_limit)
    return result
