from emnist import extract_training_samples,extract_test_samples
from keras.utils.np_utils import to_categorical
from Utils import *
from nltk.tokenize import word_tokenize
import json

def load_emnist_data():
    train_images, train_labels = extract_training_samples('byclass')
    test_images, test_labels = extract_test_samples('byclass')

    train_images = train_images/255
    test_images = test_images/255

    train_images = train_images.reshape(list(train_images.shape)+[1])
    test_images = test_images.reshape(list(test_images.shape)+[1])

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def parse_mind_line(lines,i):
    splited = lines[i].strip('\n').split('\t')
    doc_id,vert,subvert,title= splited[0:4]
    return title,vert

def load_text_data(filename,max_length,path):
            
    parse_line = parse_mind_line
    
    texts = []
    labels = []
    label_dict = CustomCounter()
    word_dict = CustomCounter(1)
    

    with open(os.path.join(path,filename)) as f:
        lines=f.readlines()
            
    for i in range(len(lines)):
        text,label = parse_line(lines,i)
        label = label_dict[label]
        text = word_tokenize(text.lower())
        #text = text.lower().split()
        text = [word_dict[text[i]] for i in range(min(len(text),max_length))]
        text = text + [0]*(max_length-len(text))
        text = text[:max_length]
        
        texts.append(text)
        labels.append(label)
    
    texts = np.array(texts)
    labels = np.array(labels)
    
    labels = to_categorical(labels)

    TRAIN_NUM = int(0.8*len(texts))
    train_data = texts[:TRAIN_NUM]
    train_labels = labels[:TRAIN_NUM]
    test_data = texts[TRAIN_NUM:]
    test_labels = labels[TRAIN_NUM:]

    return train_data, train_labels, test_data, test_labels, word_dict, label_dict