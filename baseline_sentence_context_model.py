import os
import sklearn
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import metrics
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm.classes import LinearSVC
from nltk.corpus import words

import scipy.stats
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
import re
import gensim
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.linear_model.logistic import LogisticRegression


'''                                Mean                STD
all 0's                          0.654900670438    0.0945545454328
MaxEnt
    prev_line_baseline           0.654812025117    0.0942173457949
    prev_line_lex                0.655019405117    0.0948743028013
    prev_line_syn                0.655499472833    0.0952000998847

    
prev_word:
chars/word
    (-0.033810099382043747, 0.020478890869689799)
word/sent
    (0.039439625690253596, 0.0068592461324990259)
sent-len
    (0.029123926919931077, 0.04592269475516407)

H1
(0.030500997553461369, 0.036570384400908894)
H2
(0.093578707245121337, 1.3060421522516402e-10)

prev_word
H1
(0.0458003500508165, 0.0016891436288882971)
H2
(0.03550747397494556, 0.014938293152579381)

'''

def get_feature_names():
    return [
#             'chars/word', 
#             'word/sent', 
#             'sent-len', 
#              'pron', 
#              'noun', 
#              'nnp',
#              'rb',
#              'jj',
#              'vbz',
#              'vbg',
#              'cc',
#              'punc',
#              'cd',
#             'height-tree',
#             'NP',
#             'VP',
#             'SBAR',
             'H1',
             'H2'
            ]

def get_baseline_features(text):
    features = []
    
    words = nltk.word_tokenize(text)
    
    # avg number of chars per word 
    arr = []
    for word in words:
        arr.append(len(word))
    features.append(np.mean(arr))
    
    # number of words in the sentence
    features.append(len(words))
    
    # length of sentence
    features.append(len(text))
    
    return features

def get_bagofwords_features(text, vectorizer):
    feats = vectorizer.transform([text]).toarray()[0]
    return feats
    
    

def get_lexical_features(text, pos):
    features = []
    words = pos.split(' ')

    pron_count = 0
    noun_count = 0
    nnp_count = 0
    rb_count = 0
    jj_count = 0
    vbz_count = 0
    vbg_count = 0
    cc_count = 0
    punc = 0
    cd_count = 0

    for word in words:
        tag = word.split('_')[1]

        if tag == 'PRP':
            pron_count += 1
        if tag == 'NN' or tag == 'NNS':
            noun_count += 1
        if tag == 'NNP':
            nnp_count += 1
        if tag == 'RB':
            rb_count += 1
        if tag == 'JJ':
            jj_count += 1
        if tag == 'VBZ':
            vbz_count += 1
        if tag == 'VBG':
            vbg_count += 1
        if tag == 'CC':
            cc_count += 1
        if tag in ['(', ')', ',']:
            punc += 1
        if tag == 'CD':
            cd_count += 1

    features.append(pron_count) # number of pronouns per sentence
    features.append(noun_count) # number of nouns per sentence
    features.append(nnp_count) # number of proper nouns per sentence
    features.append(rb_count)
    features.append(jj_count)
    features.append(vbz_count)
    features.append(vbg_count)
    features.append(cc_count)
    features.append(punc)
    features.append(cd_count)
    
    return features

def get_syntax_features(text, parse):
    features = []
    
    height = len(parse.split('\n'))
    np = parse.count('(NP')
    vp = parse.count('(VP')
    sbar = parse.count('(SBAR')
    
    features.append(height) # height of tree
    features.append(np) # number of noun phrase per sentence
    features.append(vp) # number of verb phrase per sentence
    features.append(sbar) # number of SBAR per sentence
    
    return features

def get_wordvector_features(text, model):
    num_features = model.syn0.shape[1]
    
    feature_vec = np.zeros((num_features,),dtype="float32")
    text = re.sub(r'[^\w\s]','',text).lower()
    words = nltk.word_tokenize(text)
    for word in words:
        if word not in stopwords.words('english'):
            if word in model:
                feature_vec = np.add(feature_vec, model[word])
    
    feature_vec = np.divide(feature_vec, len(words))
    return feature_vec

def get_prevline_features(prev_line, text, prev_pos, prev_parse):
    vals = prev_line.strip().split('\t')
    if len(vals) == 3:
        prev_text = vals[2]
        
        features = []
        features.extend(get_baseline_features(prev_text))
        features.extend(get_lexical_features(prev_text, prev_pos))
        features.extend(get_syntax_features(prev_text, prev_parse))
        features.extend(get_structure_features(vals[1], prev_text))
    else:
        features = np.zeros(19)
    
    return features
    
def get_structure_features(heading, text):
    features = []
    if heading == 'H1':
        features.append(0)
    else:
        features.append(1)
    
    if heading == 'H2':
        features.append(0)
    else:
        features.append(1)
        
    return features


def get_features(vectorizer, heading, text, pos, parse, model, prev_line, prev_pos, prev_parse):
    f = []
    f.extend(get_baseline_features(text))
    f.extend(get_lexical_features(text, pos))
    f.extend(get_syntax_features(text, parse))
    
    f.extend(get_structure_features(heading, text))
    f.extend(get_prevline_features(prev_line, text, prev_pos, prev_parse))
    
    #f.extend(get_wordvector_features(text, model))
    
    #f.extend(get_bagofwords_features(text, vectorizer))
    return f


def load_pos(source):
    pos = {}
    
    f = open('annotated_sentences_CLEANED_POS_STANFORD/BIO/%s.txt' % source, 'r')
    lines = f.readlines()
    for c, line in enumerate(lines):
        vals = line.strip().split('\t')
        pos[c] = vals[0]
    
    return pos

def load_parse_tree(source):
    parse = {}
    
    f = open('annotated_sentences_CLEANED_PARSE/BIO/%s.txt' % source, 'r')
    lines = f.readlines()
    count = 0
    str = ''
    for line in lines:
        if line.strip() == '':
            parse[count] = str
            count += 1
            str = ''
            continue
        else:
            str += line
    
    return parse

sources = [
                       #4848 BIO
            15463, 
            18000,
            17423,
            15855,
            18565,
            15461,
            15460,
            15779,
            18887,
            15456,
            ]



path = 'annotated_sentences_CLEANED/BIO/%s'

total = []

model = None
#model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


for source in sources:
    # predict this source 
    print source
    
    X_train = []
    X_test = []
    
    y_train = []
    y_test = []
    
#     vectorizer = CountVectorizer(analyzer = "word",   \
#                              stop_words = 'english',   \
#                              #max_features = 5000
#                              )
    vectorizer = None 

    sents = []
    for source2 in sources:
        if source2 != source:
            s_path = path % source2   
            f = open('%s.txt' % (s_path),'r')
            lines = f.readlines()
            for line in lines:
                vals = line.strip().split('\t')
                sents.append(vals[2])
            f.close()
          
    model = gensim.models.Word2Vec(sents)
    
    #vectorizer.fit(sents)
    
    
    
    for source2 in sources:
        
        pos = load_pos(source2)
        parse = load_parse_tree(source2)
        
        s_path = path % source2
                
        f = open('%s.txt' % (s_path),'r')
        lines = f.readlines()
        
        prev_line = "-1\t\t"
        prev_pos = None
        prev_parse = None
            
        for c, line in enumerate(lines):
            if line.strip() == '':
                break
            vals = line.strip().split('\t')
            text = vals[2]
            
            if source2 != source:
                X_train.append(get_features(vectorizer, vals[1], text, pos[c], parse[c], model, prev_line, prev_pos, prev_parse))
            
                if int(vals[0]) == 0:
                    y_train.append('0')
                else:
                    y_train.append('1')
            else:
                X_test.append(get_features(vectorizer, vals[1], text, pos[c], parse[c], model, prev_line, prev_pos, prev_parse))
                
                if int(vals[0]) == 0:
                    y_test.append('0')
                else:
                    y_test.append('1')
                    
            prev_line = line
            prev_pos = pos[c]
            prev_parse = parse[c]
    
    clf = LogisticRegression()
    
#     clf2 = LinearSVC()
#     clf2.fit(X_train, y_train)
#     
#     ch2 = SelectFromModel(clf2, prefit=True)
#     
#     X_train = ch2.transform(X_train)
#     X_test = ch2.transform(X_test)
    
    clf.fit(X_train, y_train)
    
    print len(y_train)
    print len(y_test)
    
    pred = clf.predict(X_test)
    
    #pred = ['0']* len(y_test)
    score = metrics.accuracy_score(y_test, pred)
    score = metrics.precision_recall_curve
    print("accuracy:   %0.3f" % score)
    total.append(score)
    
#     n = 20
#      
#     feature_names = get_feature_names()
#     coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
#     top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
#     for (coef_1, fn_1), (coef_2, fn_2) in top:
#         print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)                   
#                     

# y_test.extend(y_train)
# for c, i in enumerate(get_feature_names()):
#     X = []
#     for item in X_test:
#         X.append(item[c])
#     for item in X_train:
#         X.append(item[c])
#     X = np.array(X)
#     Y = np.array(y_test).astype(np.float)
#     print i
#     print scipy.stats.pearsonr(X, Y)
#     
    

print np.mean(total)
print np.std(total)
