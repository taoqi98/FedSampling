import numpy as np
from sklearn.metrics import *
import json

def local_sampling(candidate_samples,K,hatN,):
    sample_chocie = np.random.binomial(size=(len(candidate_samples),),n=1,p=K/hatN)
    candidate_samples = candidate_samples[sample_chocie==1]
    return candidate_samples


def func_flatten_weights(weights):
    a = []
    for i in range(len(weights)):
        w = weights[i].reshape((-1,))
        a.append(w)
    a = np.concatenate(a)
    return a 

def func_unflatten_weights(f_weights,old_weights):
    r_weights = []
    start = 0
    for i in range(len(old_weights)):
        ed = start + old_weights[i].reshape((-1,)).shape[0]
        lw = f_weights[start:ed]
        lw = lw.reshape(old_weights[i].shape)
        r_weights.append(lw)
        start = ed
    return r_weights


class Estimator:
    def __init__(self,train_users,alpha,M):
        self.M = M
        self.alpha = alpha
        self.train_users = train_users
        
    def query(self,userid):
        fake_response = np.random.randint(1,self.M)
        real_response = len(self.train_users[userid])
        choice = np.random.binomial(n=1,p=self.alpha)
        response = choice*real_response + (1-choice)*fake_response
        return response
    
    def estimate(self,):
        R = 0
        for uid in range(len(self.train_users)):
            R += self.query(uid)
        hat_N =  (R-len(self.train_users)*(1-self.alpha)*self.M/2)/self.alpha
        hat_N = max(hat_N,len(self.train_users))
        return hat_N

def FedSampling(alpha,M,K,r,model,train_users,train_data,train_labels,test_data,test_labels):

    estimator = Estimator(train_users,alpha,M)
    Res = []
    for i in range(8000):
        all_gradients = []
        old_weights = model.get_weights()
        flan_old_weights = func_flatten_weights(old_weights)
        hatN = estimator.estimate()
        user_indexs = np.random.permutation(len(train_users))[:int(len(train_users)*r)]
        for uid in user_indexs:
            sample_ids = local_sampling(train_users[uid],K,int(hatN*r))
            if len(sample_ids)>0:
                x,y = train_data[sample_ids], train_labels[sample_ids]
                model.train_on_batch(x,y)
                weights = model.get_weights()
                flan_weights = func_flatten_weights(weights)
                flan_gradients = flan_weights-flan_old_weights
                flan_gradients = flan_gradients*len(sample_ids)/K
                all_gradients.append(flan_gradients)
                model.set_weights(old_weights)
        gradients = np.sum(all_gradients,axis=0)
        flan_weights = flan_old_weights+gradients
        weights = func_unflatten_weights(flan_weights,old_weights)
            
        model.set_weights(weights)
        
        if i>0 and i%4==0:
            pred = model.predict(test_data).argmax(axis=-1)
            lbs = test_labels.argmax(axis=-1)
            macro_f1 = f1_score(lbs,pred,average='macro',)
            micro_f1 = f1_score(lbs,pred,average='micro')
            weight_f1 = f1_score(lbs,pred,average='weighted')
            print(i,macro_f1,micro_f1,weight_f1)
            Res.append([macro_f1,micro_f1,weight_f1])
            
    with open('FedSampling.json','a') as f:
        s = json.dumps(Res)
        f.write(s+'\n')

    return Res

def FedSamplingText(alpha,M,K,model,train_users,train_data,train_labels,title_word_embedding_matrix):
    TRAIN_NUM = len(train_labels)
    sample_to_user = np.zeros((TRAIN_NUM,),dtype='int32')
    for uid in train_users:
        for sid in train_users[uid]:
            sample_to_user[sid] = uid

    estimator = Estimator(train_users,alpha,M)
    Res = []
    for i in range(TRAIN_NUM//K):
        all_gradients = []
        old_weights = model.get_weights()
        flan_old_weights = func_flatten_weights(old_weights)
        hatN = estimator.estimate()
        samples = np.random.binomial(size=(TRAIN_NUM,),n=1,p=K/hatN)
        for uid in sample_to_user[np.where(samples>0)[0]]:
            #sample_ids = local_sampling(train_users[uid],KK,hatN)
            sample_ids = samples[train_users[uid]]
            sample_ids = train_users[uid][np.where(sample_ids==1)]
            x,y = train_data[sample_ids], train_labels[sample_ids]
            x = title_word_embedding_matrix[x]
            
            model.train_on_batch(x,y)
            weights = model.get_weights()
            flan_weights = func_flatten_weights(weights)
            flan_gradients = flan_weights-flan_old_weights
            flan_gradients = flan_gradients*len(sample_ids)/K
            all_gradients.append(flan_gradients)
            model.set_weights(old_weights)
            
        gradients = np.sum(all_gradients,axis=0)
        flan_weights = flan_old_weights+gradients
        weights = func_unflatten_weights(flan_weights,old_weights)
            
        model.set_weights(weights)