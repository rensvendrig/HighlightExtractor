import nltk.data
from data_structure import Document,DocumentCluster,Sentence

class DocumentConverter(object):
    def __init__(self,sent_converter):
        self.sent_converter=sent_converter
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    
    def convert_to_document(self,text_data,title=None):
        ''' text_data can be a string or string list
            title is a string
        '''
        if isinstance(text_data, str):
            text_data=self.sent_detector.tokenize(text_data.strip())
        sents=[]
        for content in text_data:
            sent=self.sent_converter.convert_to_sentence(content)
            if sent:
                sents.append(sent)
        doc=Document(sents)
        if title:
            doc.title=self.sent_converter.convert_to_sentence(title)
        return doc