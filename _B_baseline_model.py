import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import f1_score, precision_score, recall_score

def is_confused(c1,c2,c3,c4):
    c1 = int(c1)
    c2 = int(c2)
    c3 = int(c3)
    c4 = int(c4)
    if (c1 > 0 and c2 > 0 and c3 > 0) or (c2 > 0 and c3 > 0 and c4 > 0) or (c1 > 0 and c3 > 0 and c4 > 0) or (c1 > 0 and c2 > 0 and c4 > 0):
        return True
    return False

def create_para(head_val, page, c1, c2,c3, c4, sent):
    page = int(page)
    
    para = {
            'sents': [],
            'df': 0,
            'page': page,
            'sy': 0,
            'confuse_ct': 0,
            'eq': 0
            }
    
    add_to_para(para, head_val, page, c1,c2,c3,c4, sent)
    
    return para

def add_to_para(para, head_val, page, c1,c2,c3,c4, sent):
    para['sents'].append(sent)

    if (len(head_val) == 2 and head_val[1] == 'DF') or head_val[0] == 'DF':
        para['df'] += 1
    if (len(head_val) == 2 and head_val[1] == 'SY') or head_val[0] == 'SY':
        para['sy'] += 1
    
    if head_val[0] == 'EQ':
        para['eq'] += 1
    
    if is_confused(c1,c2,c3,c4):
        para['confuse_ct'] += 1
    
    
def add_page_info(page_info, page, type_n):
    if page not in page_info:
        page_info[page] = {'HT': 0,
                           'FIG': 0,
                           'EX': 0,
                           'PR': 0,
                           'TBL': 0,
                           'BR': 0,
                           'H1': 0,
                           'FN': 0}
    page_info[page][type_n] += 1

def merge_para_page_info(paras, pages_info):
    for para in paras:
        page = para['page']
        page_info = pages_info.get(page, {})
        para.update(page_info)

def read_file(val):
    
    list_paras = []
    
    page_info = {}
    
    file = open('annotated_sentences_CLEANED/PHYS_dedup/4_count/%s.txt' % val,'r')
    lines = file.readlines()
    
    vocab = lines[0].strip().split(',')
    
    current_para = None
    
    for line in lines[2:]:
        page, head, c1, c2, c3, c4, sent = line.strip().split('\t')
        
        head_val = head.strip().split(' ')
        if head_val[0] == 'P' and not (len(head_val) == 2 and head_val[1] == 'SM'):
            if current_para != None:
                list_paras.append(current_para)
            current_para = create_para(head_val, page, c1, c2,c3, c4, sent)
        elif head_val[0] in ['', 'DF', 'SY', 'EQ']:
            add_to_para(current_para, head_val, page, c1,c2,c3,c4,sent)
        else:
            if head_val[0] not in ['P', 'SM', 'H2', 'H3']:
                add_page_info(page_info, page, head_val[0])
            if current_para != None:
                list_paras.append(current_para)
            current_para = None
            
    merge_para_page_info(list_paras, page_info)
    return list_paras

def get_features(paras):
    X = []
    for para in paras:
        val = []
        val.append(para['page'])
        X.append(val)
    return X

def get_ys(paras):
    Y = []
    for para in paras:
        if float(para['confuse_ct'])/float(len(para['sents'])) >= .5:
            Y.append(1)
        else:
            Y.append(0)
    return Y

def create_dataset():
    paras = read_file(22)
    paras.extend(read_file(23))
    paras.extend(read_file(24))
    paras.extend(read_file(30))
    paras.extend(read_file(33))
    
    print len(paras)
    return paras

def run():
    paras = create_dataset()
    
    X = np.array(get_features(paras))
    Y = np.array(get_ys(paras))
    
    skf = StratifiedKFold(Y, n_folds=3)
    
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        clf = LogisticRegression()
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print precision_score(y_test, y_pred)
        print recall_score(y_test, y_pred)
        print f1_score(y_test, y_pred)
        
        
    
    
run()

