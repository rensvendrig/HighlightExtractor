import jsonpickle
import codecs
import nlputils.data_structure as DS

def read_from_file(filename,sent_converter=None):
    f= codecs.open(filename, encoding='utf-8')
    datasets=[]
    while True:
        line=f.readline()
        if not line: break
        line=line.strip()
        items=line.split('|||')
        if len(items)<4: continue
                
        cluster=DS.DocumentCluster([])
        datasets.append(cluster)
        cluster.models=[]
        cluster.id=items[0]
        query_indicator=items[1]
        doc_ids=items[2].split(';')
        model_ids=items[3].split(';')
        if query_indicator=='1':
            query_content=f.readline().strip()
            cluster.query=_convert_to_document(query_content)
        for doc_id in doc_ids:
            content=f.readline().strip()
            doc=_convert_to_document(content)
            doc.id=doc_id
            cluster.docs.append(doc)
        for model_id in model_ids:
            content=f.readline().strip()
            summary=_convert_to_document(content)
            summary.id=model_id
            cluster.models.append(summary)
    f.close()
    return datasets

def _convert_to_document(content,sent_converter=None):
    sents=[]
    for sent_content in content.split('|||'):
        if not sent_converter:
            sent=DS.Sentence(sent_content)
        else:
            sent=sent_converter.convert_to_sentence(sent_content)
        sents.append(sent)
    doc=DS.Document(sents)
    return doc

def main():
    filename=r'D:\Experiment\Data\SummarizationDataset\duc01.common'
    datasets=read_from_file(filename)
    save_filename=r'D:\Experiment\Data\SummarizationDataset\duc01.pyjson'
    f=codecs.open(save_filename, 'w', encoding='utf-8')
    for cluster in datasets:
        json_str=jsonpickle.encode(cluster)
        f.write(json_str)
        f.write('\n')
    f.close()
    return

if __name__=='__main__':
    main()