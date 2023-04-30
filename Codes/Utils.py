import os
import numpy as np

class CustomCounter(dict):
    def __init__(self,index=0):
        self.index = index
        super().__init__()
        
    def __getitem__(self,key):
        self.push(key)
        return super().__getitem__(key)

    def push(self,key):
        if self.__contains__(key):
            return False
        self.__setitem__(key, self.index)
        self.index += 1
        return True

def load_matrix(embedding_path,word_dict):
    embedding_matrix = np.zeros((len(word_dict)+1,300))
    have_word=[]
    with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index]=np.array(tp)
                have_word.append(word)
    return embedding_matrix