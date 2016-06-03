from pysumm.factory import build_common_pipeline
from nlputils.factory import build_common_document_converter
from nlputils.data_structure import DocumentCluster
from pysumm.rouge_wrap import RougeWrapper
from PyTimer import Stopwatch

def end2end_test(doc_content_list,query_content,human_summary_content):
    doc_converter=build_common_document_converter()
    pipeline=build_common_pipeline()
    doc_list=[]
    for content in doc_content_list:
        doc=doc_converter.convert_to_document(content)
        doc_list.append(doc)
    doc_cluster=DocumentCluster(doc_list)
    query=doc_converter.convert_to_document(query_content)
    summary=pipeline.summarize(doc_cluster, query=query, length_mode=1, length_limit=4)
    
    for sent in summary.sents:
        print sent.saliency,'\t',sent.content  
        
    system_summaries=[summary.content]
    human_summaries=[[human_summary_content]] 
    rouge=RougeWrapper()
    result=rouge.evaluateFromString(system_summaries,human_summaries,result_fold=r'D:\temp')
    for k,v in result.items():
        print k,'\t',v
    return


def main():
    doc1_text='''
Geraldine Largay knew she was so very lost that the chances of her making it out of the thick Maine woods were gone.
She had been writing in her journal every day and one of her final entries showed she was resigned to her fate.
"When you find my body please call my husband George and my daughter Kerry. It will be the greatest kindness for them to know that I am dead and where you found me -- no matter how many years from now. Please find it in your heart to mail the contents of this bag to one of them."
It was August 6, 2013, about 15 days after she left the Appalachian Trail to use the bathroom.
Her body was found more than two years later, in a sleeping bag inside a zipped up tent.
    '''
    doc2_text='''
Investigators this week released documents and photos related to the case. Texts and journal entries reveal Largay, 66, was alive for almost a month after she went missing.
Largay tried on July 22 to text her husband, who was meeting her at certain points along the almost 2,200-mile long trail, to get help from the Appalachian Mountain Club.
"In somm trouble. Got off trail to go to br. Now lost. Can u call AMC to c if a trail maintainer can help me. Somewhere north of woods road. Xox," she wrote.
But the text didn't go through. She walked west and kept trying to send the message. Ten more attempts, the last one at 12:25 p.m.
She tried to send a blank message two hours later. It didn't go through.
She tried another message the next day, writing, "Lost since yesterday. Off trail 3 or 4 miles. Call police for what to do pls. Xox." It also failed and she tried one more time to no avail.
By this time, Largay had set up her tent nearly 2 miles from the trail. She had some food, water and her camping supplies. And a journal that she wrote in each day.
    '''
    doc3_text='''
One of the Maine wardens who compiled the evidence wrote that Largay's writings were personal letters to her family.
There were entries through August 10 then nothing until the 18th. It was the last entry, 27 days after she got lost.
A forester on contract with the U.S. Navy found her campsite on October 11, 2015.
    '''
    query_content='''
    Her husband told wardens that Gerry Largay used her cell phone sparingly, depending on the circumstances.
    '''
    query_content='''
    The text of July 23 wasn't the last Largay tried to send. There were also messages on the afternoon of July 30. And two texts were deleted on August 6, the same date of her hopeless journal entry.
    '''
    human_summary='''
    There were entries through August 10 then nothing until the 18th. It was the last entry, 27 days after she got lost.
    '''
    stopwatch=Stopwatch()
    stopwatch.start()
    end2end_test((doc1_text,doc2_text,doc3_text),query_content,human_summary)
    stopwatch.end_and_show('Summarizer')    
    return


if __name__=='__main__':
    main()
