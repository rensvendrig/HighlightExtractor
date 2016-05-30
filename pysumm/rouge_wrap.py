'''
Created on May 5, 2016

@author: Ziqiang
'''
import os
import codecs
import logging
import xml.etree.cElementTree as ET

class RougeWrapper(object):
    '''
    classdocs
    '''
    def __init__(self, rouge_path='D:\\Experiment\\Tools\\rouge'):
        '''
        Constructor
        '''
        common_args=' -n 2 -u -c 95 -x -r 1000 -f A -p 0.5 -t 0'
        if not os.path.exists(rouge_path):
            logging.error('Can not find ROUGE in path:{0}'.format(rouge_path))
            exit(1)
        self.args=' '.join([common_args,'-e',os.path.join(rouge_path,'data')])
        self.rouge_script=os.path.join(rouge_path,'ROUGE-1.5.5.pl')
    
    def evaluateFromString(self,system_summaries,model_list_summaries,result_fold,evaluation_ids=None,length_limint=None):
        temp_fold=os.path.join(result_fold,'temp')
        if not os.path.exists(temp_fold):
            os.makedirs(temp_fold)
        system_files=[]
        model_list_files=[]
        for index in xrange(len(system_summaries)):
            eid=str(index+1)
            if evaluation_ids: eid=evaluation_ids[index]
            system_file=os.path.join(temp_fold,'peer.'+eid+'.txt')
            system_files.append(system_file)
            with codecs.open(system_file, 'w', encoding='utf-8') as f:
                f.write(system_summaries[index])
            model_files=[]
            model_list_files.append(model_files)
            for model_index,model_summary in enumerate(model_list_summaries[index]):
                model_file=os.path.join(temp_fold,'model.'+chr(model_index+ord('A'))+'.'+eid+'.txt')
                model_files.append(model_file)
                with codecs.open(model_file, 'w', encoding='utf-8') as f:
                    f.write(model_summary)
        return self.evaluateFromFile(system_files, model_list_files, result_fold, evaluation_ids, length_limint)
    
    def evaluateFromFile(self,system_files,model_list_files,result_fold,evaluation_ids=None,length_limint=None):
        peer_id='1'
        if not os.path.exists(result_fold):
            os.makedirs(result_fold)
        root=ET.Element('ROUGE_EVAL')
        root.set('version','1.0')
        for index in range(len(system_files)):
            system_file=system_files[index]
            model_first_file=model_list_files[index][0]
            eval_item=ET.SubElement(root,'EVAL')
            eid=str(index+1)
            if evaluation_ids: eid=evaluation_ids[index]
            eval_item.set('ID',eid)
            ET.SubElement(eval_item,'PEER-ROOT').text=os.path.dirname(system_file)
            ET.SubElement(eval_item,'MODEL-ROOT').text=os.path.dirname(model_first_file)
            ET.SubElement(eval_item,'INPUT-FORMAT').set('TYPE','SPL')
            peer_root=ET.SubElement(eval_item,'PEERS')
            peer_item=ET.SubElement(peer_root,'P')
            peer_item.text=os.path.basename(system_file)
            peer_item.set('ID',peer_id)
            model_root=ET.SubElement(eval_item,'MODELS')
            for model_id, model_file in enumerate(model_list_files[index]):
                model_item=ET.SubElement(model_root,'M')
                model_item.set('ID',str(model_id+1))
                model_item.text=os.path.basename(model_file)
        tree=ET.ElementTree(root)
        config_filename=os.path.join(result_fold,'config.xml')
        tree.write(config_filename)
        rouge_args=self.args
        if length_limint:
            rouge_args+=' '+length_limint
        #rouge_args+=' -a'
        rouge_args+=' '+config_filename
        rouge_args+=' '+peer_id
        logging.info('ROUGE parameters:'+rouge_args)
        #p=subprocess.Popen(["perl", self.rouge_script,rouge_args,config_filename,peer_id], stdout=subprocess.PIPE,shell=True)
        #output,err=p.communicate()
        result_filename=os.path.join(result_fold,'rouge_result.txt')
#         with open(result_filename,'w') as f:
#             f.write(output)
#         logging.info('output:'+output)
#         if err:
#             logging.error('err:'+err)
        os.system('perl '+self.rouge_script+rouge_args+' >'+result_filename)                                
        return self.analyze_output(result_filename)
    
    def analyze_output(self,result_filename):
        output={}
        with open(result_filename) as f:
            for line in f:
                items=line.split()
                if len(items)!=8: continue
                output[items[1]+' '+items[2]]=float(items[3])                                 
        return output
    
    
if __name__ == '__main__':
    system_summaries=['a b c d','a d f c']
    model_list_summaries=[['a c e','b f a'],['c d a','f s b']]
    r=RougeWrapper()
    r.evaluateFromString(system_summaries, model_list_summaries, result_fold=r'D:\Experiment\Result\SciSumm\temp')