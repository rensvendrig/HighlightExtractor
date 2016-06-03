import time

class Stopwatch(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def start(self):
        self.__start_time=time.time()
        
    def end(self):
        self.__end_time=time.time()
        
    def show_interval(self,label):
        elapse=self.__end_time-self.__start_time
        print 'Run time for: {0} is {1:.2f}s'.format(label, elapse)
        
    def end_and_show(self,label):
        self.end()
        self.show_interval(label)