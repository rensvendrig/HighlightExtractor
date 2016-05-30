
class Word(object):
    '''
        Used to hold attributes of a token, such as stem, pos, ner etc.
    '''
    def __init__(self,content):
        self.content=content
        return

class Sentence(object):
    def __init__(self,content):
        self.content=content
        return


class Document(object):
    def __init__(self,sents):
        self.sents=sents
        return
    
    @property
    def content(self):
        return '\n'.join([sent.content for sent in self.sents])
    
class DocumentCluster(object):
    def __init__(self,docs):
        self.docs=docs
        return
    
    @property
    def total_sentences(self):
        sents=[]
        for doc in self.docs:
            sents+=doc.sents
        return sents