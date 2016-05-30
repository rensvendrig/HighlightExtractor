import sentence_analyser as sa
from doc_utils import DocumentConverter
from nltk.corpus import stopwords

def build_common_sentence_converter():
    formatters=[]
    formatters.append(sa.SplitFormatter())
    processors=[]
    processors.append(sa.Tokenizer())
    processors.append(sa.Stemmer())
    stop=stopwords.words('english')
    processors.append(sa.TFCounter(stopwords=stop))
    sent_converter=sa.SentenceConverter(formatters,processors)
    return sent_converter

def build_common_document_converter():
    sent_converter=build_common_sentence_converter()
    doc_converter=DocumentConverter(sent_converter)
    return doc_converter

