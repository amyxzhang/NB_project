import os
import sklearn
import enchant
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import metrics
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm.classes import LinearSVC
from nltk.corpus import words as nltk_words

import scipy.stats
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
import re
import gensim
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.linear_model.logistic import LogisticRegression

py_di = enchant.Dict("en_US")

'''                                Mean                STD
all 0's                          0.654900670438    0.0945545454328
MaxEnt
    bag_of_words                 0.618716061744    0.0625846486785          
    baseline                     0.652874134316    0.0869648106803
    lexical                      0.652495797057    0.08436279033
    syntax                       0.65629050683     0.0912282601797
    lex+syn                      0.651897129605    0.0836080498923
    base+lex+syn                 0.650313070006    0.0838769208073
    bag_of_words+base+lex+syn    0.611539551137    0.0541640654762
    
    word_embed(goog)             0.655030140379    0.0949038649832
    word_embed_g+base+lex_syn    0.651912561317    0.0867175187676
    word_embed(self)             0.654301868043    0.0939389124032
    word_embed_s+base+lex+syn    0.651244844912    0.0849562312079


Pearson features:
chars/word
    (-0.032841873597825748, 0.024382085692868444)
word/sent
    (0.13300820881390693, 5.4398564470751635e-20)
sent-len
    (0.12881284303815255, 7.7413361366877243e-19)
    
    
pron
    (0.022155019177976101, 0.12893106509222269)
noun
    (0.13244443040014775, 7.8115075741689847e-20)
nnp
    (-0.010146727575506982, 0.48686142231108553)
rb
    (0.056885706479588953, 9.5725243767494237e-05)
jj
    (0.069524452886557592, 1.8436658905048075e-06)
vbz
    (0.071265982313866091, 1.0102678096895345e-06)
vbg
    (0.039298612089992267, 0.0070615493254230794)
cc
    (0.065144224387079466, 7.8706903269683464e-06)
punc
    (0.071013389357245188, 1.1033260697539058e-06)
cd
    (0.051338849247170866, 0.00043113980578911814)
    
    
    
height-tree
    (0.11609324428898928, 1.4381438807378656e-15)
NP
    (0.11517681312737664, 2.3999091767636427e-15)
VP
    (0.093260577114725449, 1.5084003215251578e-10)
SBAR
    (0.040674054948942516, 0.0052986875334617529)
    
word_dict
(0.13485526192397562, 1.6443735967042806e-20)
(0.12469158451877489, 9.6682952154116841e-18)


technical terms versus any new word
take another corpus with common words

identifying terminology

send the sentence and the comment


two terms used in an unusual configuration - defined both but used together

distance from previous mention of technical term, distance from definition

towards end of chapter, more comments?

problem and example vs definition? which has more questions.

using "it" not knowing what to refer to

make claims, but student doesn't understand 


look at terms which don't appear in google corpus

snowball..? take many papers and then identify several patterns. 

how to tell what is definition? find list of potential terms. 

or assume each occurrence is definition. and train a classifier. and check 20 occurrences.



use two datasets?


'''

def get_feature_names():
    return [
            'chars/word', 
            'word/sent', 
            'sent-len', 
             'pron', 
             'noun', 
             'nnp',
             'rb',
             'jj',
             'vbz',
             'vbg',
             'cc',
             'punc',
             'cd',
            'height-tree',
            'NP',
            'VP',
            'SBAR',
                'word_dict_0',
                'word_dict_1',
                'word_dict_2',
                'word_dict_3',
            'word_ct_0',
            'word_ct_1',
            'word_ct_2',
            'word_ct_3',
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

def get_word_dict_features(c, text, pos, word_dict):
    f = []
    
    log_c = np.sqrt(c)
    
    nn_count = 0
    count = 0
    
    no_c_count = 0
    for word in pos.split(' '):
        val = word.split('_')
        vv = val[0].lower()
        #if (val[1] == 'NN' or val[1] == 'NNS'):
        nn_count += 1
        if vv not in word_dict:
            count += 5.0*log_c
            no_c_count += 5.0
            
            word_dict[vv] = 1
        else:
            count += float((1.0*log_c)/word_dict[vv])
            no_c_count += float((1.0)/word_dict[vv])
    if nn_count != 0:
        f.append(count/nn_count)
    else:
        f.append(0)
    
    f.append(count)
    
#     if nn_count != 0:
#         f.append(no_c_count/nn_count)
#     else:
#         f.append(0)
    
    f.append(no_c_count)

    return f

def word_counts_feature(word_counts, text):
    total = 0
    wc_total = 0
    features = []
    words = nltk.word_tokenize(text)
    for word in words:
        if word_counts.get(word, 0) < 10:
            total += 1.0
        wc_total += word_counts.get(word, 0)

    features.append(total)
    features.append(wc_total)
    
    features.append(total/len(words))
    features.append(wc_total/len(words))
    
    
    return features

def get_features(c, vectorizer, text, pos, parse, model, word_counts, word_dict):
    f = []
    
    f.extend(get_baseline_features(text))
    f.extend(get_lexical_features(text, pos))
    f.extend(get_syntax_features(text, parse))
    
    f.extend(get_word_dict_features(c, text, pos, word_dict))
    
    f.extend(word_counts_feature(word_counts, text))
    
    f.extend(get_wordvector_features(text, model))
    
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
total2 = []

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
    pred_sents = []
    
    word_counts = {}
    for source2 in sources:
        if source2 != source:
            s_path = path % source2   
            f = open('%s.txt' % (s_path),'r')
            lines = f.readlines()
            for line in lines:
                vals = line.strip().split('\t')
                sents.append(vals[2])
                words = nltk.word_tokenize(vals[2])
                for word in words:
                    if word not in word_counts:
                        word_counts[word] = 0
                    word_counts[word] += float(1.0/len(words))
        else:
            s_path = path % source2   
            f = open('%s.txt' % (s_path),'r')
            lines = f.readlines()
            for line in lines:
                vals = line.strip().split('\t')
                pred_sents.append(vals[2])
            
        f.close()
          
    model = gensim.models.Word2Vec(sents)
    
    #vectorizer.fit(sents)
    
    
    for source2 in sources:
        
        word_dict = {}
        
        pos = load_pos(source2)
        parse = load_parse_tree(source2)
        
        s_path = path % source2
                
        f = open('%s.txt' % (s_path),'r')
        lines = f.readlines()
            
        for c, line in enumerate(lines):
            if line.strip() == '':
                break
            vals = line.strip().split('\t')
            text = vals[2]
            
            if source2 != source:
                
                
                
                X_train.append(get_features(c, vectorizer, text, pos[c], parse[c], model, word_counts, word_dict))
            
                if int(vals[0]) == 0:
                    y_train.append(0)
                else:
                    y_train.append(1)
            else:
                X_test.append(get_features(c, vectorizer, text, pos[c], parse[c], model, word_counts, word_dict))
                
                if int(vals[0]) == 0:
                    y_test.append(0)
                else:
                    y_test.append(1)
    
    
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
    
    #pred = [0]* len(y_test)
    score = metrics.accuracy_score(y_test, pred)
    prec = metrics.precision_score(y_test, pred)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)
    print("accuracy:   %0.3f   prec: %0.3f   recall: %0.3f   f1: %0.3f" % (score, prec, recall, f1))
    total.append(score)
    total2.append(f1)
    
    file2 = open('results/%s-1' % source, 'w')
    file3 = open('results/%s-0' % source, 'w')
    for s, (y, x) in zip(pred_sents, zip(pred, X_test)):
        if y == 1:
            file2.write(s + '\n')
            file2.write(str(x) + '\n')
        else:
            file3.write(s + '\n')
            file3.write(str(x) + '\n')
            
    
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
    

print 'Avg acc: %s' % np.mean(total)
print 'STD acc: %s' % np.std(total)

print 'Avg F1: %s' % np.mean(total2)
print 'Std F1: %s' % np.std(total2)
