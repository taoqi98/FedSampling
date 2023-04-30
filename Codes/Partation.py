import numpy as np

def class_noniid(train_labels,local_sample_num):
    train_users = {}
    index = train_labels.argmax(axis=-1).argsort()

    for i in range(int(np.ceil(len(train_labels)//local_sample_num))):
        s,ed = local_sample_num*i, (i+1)*local_sample_num
        ed = min(ed,len(train_labels))
        train_users[i] = index[s:ed]

    return train_users


def size_noniid(train_labels,avg,std):
    TRAIN_NUM = len(train_labels)
    r = np.random.lognormal(0,std,size=(TRAIN_NUM,))
    r = avg*r/np.ceil(r).mean()
    r = np.ceil(r)

    random_index = np.random.permutation(TRAIN_NUM)
    g = []
    train_users = {}
    ct = 0
    user_index = 0
    ix = 0
    while ct < TRAIN_NUM:
        n = int(r[ix])
        ix += 1
        if n +ct>=TRAIN_NUM:
            n = TRAIN_NUM-ct
        g.append(n)
        train_users[user_index] = random_index[ct:ct+n]
        ct += n

        user_index += 1
    return train_users