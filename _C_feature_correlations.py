import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import f1_score, precision_score, recall_score,\
    accuracy_score
from sklearn.ensemble.forest import RandomForestClassifier,\
    RandomForestRegressor
from sklearn.svm.classes import LinearSVC
from nltk.tokenize import word_tokenize
import re
import scipy.stats
from scipy.stats import pearsonr
import copy
from nltk.corpus import stopwords
from _collections import defaultdict
from sklearn.metrics.regression import r2_score
    
number_words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", 
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion", "trillion",]

liwc = {}

POS_sents = None
line_ct = 0

word_count = {}

def load_liwc(word):
    liwc[word] = []
    f1 = open('LIWC_%s.txt' % word,'r')
    lines = f1.readlines()
    for line in lines:
        item = line.strip()
        if item[len(item)-1] == '*':
            item = r'\b%s.*?\b' % item[0:len(item)-1]
        else :
            item = r'\b%s\b' % item
        
        liwc[word].append(re.compile(item))
    

def is_confused(c1,c2,c3,c4):
    c1 = int(c1)
    c2 = int(c2)
    c3 = int(c3)
    c4 = int(c4)
    if (c1 > 0 and c2 > 0 and c3 > 0) or (c2 > 0 and c3 > 0 and c4 > 0) or (c1 > 0 and c3 > 0 and c4 > 0) or (c1 > 0 and c2 > 0 and c4 > 0):
        return True
    return False

def create_para(head_val, page, c1, c2,c3, c4, sent, first_page, vocab_occur, prev_para, non_vocab_occur):
    page = int(page)
    
    para = {
            'sents': [],
            'pos_sents': [],
            'word_ct': [],
            'df': 0,
            'page': page,
            'sy': 0,
            'confuse_ct': 0,
            'eq': 0,
            'page_num': page - first_page,
            'vocab': copy.deepcopy(vocab_occur),
            'prev_para': prev_para,
            'first_occur': []
            }
    
    add_to_para(para, head_val, page, c1,c2,c3,c4, sent, non_vocab_occur)
    
    return para

def add_to_word_ct(pos_sent, para, non_vocab_occur):
    global word_count
    
    for word in pos_sent.split(' '):
        pos = word.split('_')
        l_word = pos[0].lower()
        if pos[1] in ['NN', 'NNS', 'JJ'] and len(l_word) > 1 and l_word not in stopwords.words('english'):
            
            if l_word not in non_vocab_occur:
                para['first_occur'].append(l_word)
                non_vocab_occur[l_word] = 1
            else:
                non_vocab_occur[l_word] += 1
            
            if l_word not in word_count:
                word_count[l_word] = 0
            word_count[l_word] += 1

def add_to_para(para, head_val, page, c1,c2,c3,c4, sent, non_vocab_occur):
    global line_ct
    global POS_sents
    
    para['sents'].append(sent)
    para['pos_sents'].append(POS_sents[line_ct])
    
    add_to_word_ct(POS_sents[line_ct], para, non_vocab_occur)
    
    para['word_ct'].append(len(word_tokenize(sent)))

    if (len(head_val) == 2 and head_val[1] == 'DF') or head_val[0] == 'DF':
        para['df'] += 1
    if (len(head_val) == 2 and head_val[1] == 'SY') or head_val[0] == 'SY':
        para['sy'] += 1
    
    if head_val[0] == 'EQ':
        para['eq'] += 1
    
    if is_confused(c1,c2,c3,c4):
        para['confuse_ct'] += 1
    
    
def add_page_info(page_info, page, type_n):
    p_int = int(page)
    if p_int not in page_info:
        page_info[p_int] = {'HT': 0,
                           'FIG': 0,
                           'EX': 0,
                           'PR': 0,
                           'TBL': 0,
                           'BR': 0,
                           'H1': 0,
                           'FN': 0}
    page_info[p_int][type_n.strip()] += 1

def merge_para_page_info(paras, pages_info):
    for para in paras:
        page = para['page']
        page_info = pages_info.get(page, None)
        if not page_info:
            page_info = {'HT': 0,
                           'FIG': 0,
                           'EX': 0,
                           'PR': 0,
                           'TBL': 0,
                           'BR': 0,
                           'H1': 0,
                           'FN': 0}
        para.update(page_info)

def read_file(val):
    
    global line_ct
    
    list_paras = []
    
    page_info = {}
    
    file = open('annotated_sentences_CLEANED/PHYS_dedup/4_count/%s.txt' % val,'r')
    lines = file.readlines()
    
    vocab_phrases = lines[0].strip().split(',')
    
    vocab_occur = {}
    for v in vocab_phrases:
        vocab_occur[v.strip()] = {'instance': 0,
                                  'last_occur': -1,
                                  'first_occur': -1}
    
    current_para = None
    
    non_vocab_occur = {}
    
    first_page = int(lines[2].strip().split('\t')[0])
    
    for line in lines[2:]:
        page, head, c1, c2, c3, c4, sent = line.strip().split('\t')
                
        
        head_val = head.strip().split(' ')
        if head_val[0] == 'P' and not (len(head_val) == 2 and head_val[1] == 'SM'):
            if current_para != None:
                list_paras.append(current_para)
                current_para = create_para(head_val, page, c1, c2,c3, c4, sent, first_page, vocab_occur, list_paras[-1], non_vocab_occur)
            else:
                current_para = create_para(head_val, page, c1, c2,c3, c4, sent, first_page, vocab_occur, None, non_vocab_occur)
            line_ct += 1
        elif head_val[0] in ['', 'DF', 'SY', 'EQ']:
            add_to_para(current_para, head_val, page, c1,c2,c3,c4,sent, non_vocab_occur)
            line_ct += 1
        else:
            if head_val[0] not in ['P', 'SM', 'H2', 'H3']:
                add_page_info(page_info, page, head_val[0])
            if current_para != None:
                list_paras.append(current_para)
            current_para = None
            
        for v in vocab_phrases:
            if re.search(r'\b%s\b' % v.strip(), sent.lower()):
                if vocab_occur[v.strip()]['instance'] == 0:
                    vocab_occur[v.strip()]['first_occur'] = int(page)
                vocab_occur[v.strip()]['instance'] += 1
                vocab_occur[v.strip()]['last_occur'] = int(page)
        
    if current_para != None:
        list_paras.append(current_para)
       

                
    
    merge_para_page_info(list_paras, page_info)
    return list_paras

def para_content_features(val, para):
    # Number of Summaries in the paragraph
    val.append(para['sy'])
    
    # Number of Definitions in the paragraph
    val.append(para['df'])
    
    # Is the para part of a list
    is_list = False
    for sent in para['sents']:
        if re.search(r'[0-9]\. ', sent):
            is_list = True
    val.append(is_list)
    

def page_structure_features(val, para):    
#     # Number of Examples in the page
    val.append(para['EX'])
#     
#     # Number of Figures in the page
    val.append(para['FIG'])
#     
#     # Number of Checkpoints in the page
    val.append(para['HT'])
#     
#     # Number of H1 headers in the page (start of subchapter)
    val.append(para['H1'])
#     
#     # Number of tables in the page
    val.append(para['TBL'])
#     
#     # Number of procedures in the page
    val.append(para['PR'])
#     
#     # Number of brown boxes in the page (side box?)
    val.append(para['BR'])
#     
#     # Number of footnotes in the page
    val.append(para['FN'])
    
    # page in the chapter
    val.append(para['page_num'])
    
    # page in the pdf
    val.append(para['page'])
    
    if para['prev_para'] == None:
        val.append(1)
    else:
        val.append(0)
    
    

def count_len_features(val, para):
    # Number of sentences in the para
    val.append(len(para['sents']))
    
    # Avg number of words per sentence
    val.append(np.mean(para['word_ct']))

    # Avg length of sentence
    count = []
    for sent in para['sents']:
        count.append(len(sent))
    val.append(np.mean(count))
    
    # Avg number of chars per word
    count = []
    for sent in para['sents']:
        for word in word_tokenize(sent):
            count.append(len(word))
    val.append(np.mean(count))

def equation_features(val, para):
    
    # Number of standalone equations in the paragraph
    val.append(para['eq'])
    
    # Norm number of total equations in the paragraph
    count = 0
    for sent in para['sents']:
        if '<equation>' in sent:
            count += 1
    val.append(float(count)/float(len(para['sents'])))
    
    # Number of variables in the paragraph
    count = 0
    for sent in para['sents']:
        if '<var>' in sent:
            count += 1
    val.append(count)
    
    # Number of numbers in the paragraph
    count = []
    for sent, w in zip(para['sents'], para['word_ct']):
        prev_word = None
        c = 0
        for word in word_tokenize(sent):
            if word.isdigit() and prev_word not in ["step", "Chapter", "(", "page", "Figure", "Example", "Eqs.", "Table", "Eq."]:
                c += len(word)
            prev_word = word
            
        for num in number_words:
            c += sent.count(num)
            
        count.append(float(c)/float(w))
    val.append(np.mean(count))
    
    
    # Number of <value> items
    count = 0
    for sent in para['sents']:
        count += sent.count('<value>')
    val.append(float(count)/float(len(para['sents'])))

    
def vocab_words_features(val, para):
    
    sent_text = ' '.join(para['sents'])
    
    # number of vocab words per sentence in the paragraph
    count = 0
    for sent, w in zip(para['sents'], para['word_ct']):
        for vocab in para['vocab'].keys():
            if re.search(r'\b%s\b' % vocab, sent.lower()):
                count += 1.0/float(len(vocab))
    val.append(count)
    
    # number of times para has first occurence of vocab word (probably not different than DF feature)
    count = []
    for v in para['vocab']:
        found = False
        for sent in para['sents']:
            if not found and para['vocab'][v]['instance'] == 0 and re.search(r'\b%s\b' % v, sent):
                count.append(len(re.findall(r'\b%s\b' % v, sent_text)))
                found = True
                break
    val.append(np.mean(count) if count != [] else 0.0)
    
    # avg number of times each vocab word in para has been seen so far
    count = []
    for v in para['vocab']:
        found = False
        for sent in para['sents']:
            if not found and re.search(r'\b%s\b' % v, sent):
                count.append(para['vocab'][v]['instance'])
                found = True
                break
    val.append(np.mean(count) if count != [] else 0.0)
    
    
    
    # Max Num pages since last occurence
    count = 0
    for v in para['vocab']:
        if re.search(r'\b%s\b' % v, sent_text.lower()) and para['vocab'][v]['instance'] > 0:
            x = para['page'] - para['vocab'][v]['last_occur']
            if x > count:
                count = x
    val.append(count)
    
    # Max Num pages since first occurence
    count = []
    for v in para['vocab']:
        if re.search(r'\b%s\b' % v, sent_text.lower()) and para['vocab'][v]['instance'] > 0:
            x = para['page'] - para['vocab'][v]['first_occur']
            count.append(float(x)/float(para['page_num'] + 1.0))
    val.append(np.mean(count) if count != [] else 0.0)
               
        
    
    
def _liwc_feature(feature, para):
    count = 0
    for item in liwc[feature]:
        for sent, w in zip(para['sents'], para['word_ct']):
            count += float(len(re.findall(item, sent.lower())))
    return count

def sentiment_of_words(val, para):
    
    # Occurrence of LIWC affect words
#     val.append(_liwc_feature('affect', para))
# # # #     
# # # #     # Occurrence of LIWC cogmech words
#     val.append(_liwc_feature('cogmech', para))
# # # #     
# # # #     # Occurrence of LIWC tentative words
#     val.append(_liwc_feature('tentative', para))
# # # # 
# # # #     # Occurrence of LIWC percept words
#     val.append(_liwc_feature('percept', para))
    
#     # Asking reader to remember something
    count = 0
    for sent in para['sents']:
        sent = sent.lower()
        count += sent.count('remember')
        count += sent.count('recall')
    val.append(count)
#     
    count = 0
    for sent in para['sents']:
        sent = sent.lower()
        count += sent.count('next page')
        count += sent.count('next section')
        count += sent.count('next chapter')
    val.append(count)
    
    # remarkable and surprising
    count = 0
    for sent in para['sents']:
        sent = sent.lower()
        count += sent.count('remarkable')
        count += 1 if re.search(r'surprising\b', sent) else 0
    val.append(count)
    
    # suggest
    count = 0
    for sent in para['sents']:
        sent = sent.lower()
        count += sent.count('suggest')
        count += sent.count('lead us to conclude')
        count += sent.count('leads us to conclude')
        count += sent.count('we can conclude')
    val.append(count)
    
    #doing things
    count = 0
    for sent in para['sents']:
        count += sent.count('Note ')
        count += sent.count(', note ')
        count += sent.count('Complete ')
        count += sent.count(', complete ')
        count += sent.count('work out')
        count += sent.count('working out')
    val.append(count)
    
    # Occurrence of !
    count = 0
    for sent in para['sents']:
        count += sent.count('!')
    val.append(count)
    
    # Occurrence of ?
    count = 0
    for sent in para['sents']:
        count += sent.count('?')
    val.append(count)
    
    count = 0
    num = 0
    for sent in para['sents']:
        count += sent.count('it')
        num += len(word_tokenize(sent))
    val.append(float(count)/float(num))
    
    
def example_words(val, para):
    
    sent_text = ' '.join(para['sents'])
    
    # occurrence of example phrases
    if "examples" in sent_text:
        val.append(1)
    else:
        val.append(0)
        
    if 'for example' in sent_text.lower() or 'for instance' in sent_text.lower():
        val.append(1)
    else:
        val.append(0)
        
    #yous
    count = 0
    for sent in para['sents']:
        sent = sent.lower()
        count += sent.count('you ')
        count += sent.count('your ')
    val.append(count)
    
def non_vocab_word_features(val, para):
    global word_count
    
    count = []
    for sent in para['pos_sents']:
        c = 0
        for word in sent.split(' '):
            pos = word.split('_')
            l_word = pos[0].lower()
            if pos[1] in ['NN', 'NNS', 'JJ'] and len(l_word) > 1 and l_word not in stopwords.words('english'):
                c += word_count[l_word]
        count.append(float(c)/float(len(sent.split(' '))))
    val.append(np.mean(count))
    
    print para['sents']
    print para['first_occur']
    val.append(float(len(para['first_occur']))/float(np.sum(para['word_ct'])))
    
def pos_features(val, para):
    
    count = 0 
    num = 0
    for sent_pos in para['pos_sents']:
        for word in sent_pos.split(' '):
            pos = word.split('_')[1]
            if pos == 'JJ':
                count += 1
            num += 1
    val.append(float(count)/float(num))
    
    count = 0 
    num = 0
    for sent_pos in para['pos_sents']:
        for word in sent_pos.split(' '):
            pos = word.split('_')[1]
            if pos == 'CD':
                count += 1
            num += 1
    val.append(float(count)/float(num))

    count = 0 
    num = 0
    for sent_pos in para['pos_sents']:
        for word in sent_pos.split(' '):
            pos = word.split('_')[1]
            if pos == 'NN' or pos == 'NNS':
                count += 1
            num += 1
    val.append(float(count)/float(num))
    
def get_feature_names():
    return [
            'SY',
            'DN',
            'list',
            'EX',
            'FIG',
            'HT',
            'H1',
            'TBL',
            'PR',
            'BR',
            'FN',
            'page',
            'page_pdf',
            'first_para',
            'EQ',
            'equat',
            'var',
            'num',
            'value',
#             'affect',
#             'cogmech',
#             'tentative',
#             'percept',
            'recall',
            'next',
            'remarkable',
            'suggest',
            'do things',
            '!',
            '?',
            'it',
            'examples',
            'for example',
            'yous',
            'vocab_words_per_sent',
            'number of times para has first occurence of vocab word',
            'avg # of times each vocab word in para has been seen so far',
            'Max Num pages since last occurence of a vocab word',
            'Max Num pages since FIRST occurence of a vocab word',
            'num sent',
            'avg words/sent',
            'avg len sent',
            'avg chars/word',
            'JJ',
            'CD',
            'NN/s',
            'avg word count',
            'num first occur non vocab',
            ]

    
def get_features(paras):
    X = []
    for para in paras:
        val = []
        para_content_features(val, para)
        page_structure_features(val, para)
        equation_features(val, para)
        sentiment_of_words(val, para)
        example_words(val, para)
        vocab_words_features(val, para)
        count_len_features(val, para)
        
        pos_features(val, para)
        
        non_vocab_word_features(val, para)
        
        
        X.append(val)
    return X

def get_ys(paras):
    Y = []
    for para in paras:
        if float(para['confuse_ct'])/float(len(para['sents'])) >= .3:
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
    
    sents = []
    for para in paras:
        sents.append(' '.join(para['sents']))
        
    print len(paras)
    return paras, sents


def write_sentences(paras):
    f = open('annotated_sentences_CLEANED/PHYS_dedup/4_count/sentences.txt','w')
    for para in paras:
        for sent in para['sents']:
            f.write('%s\n' % sent)
        f.write('\n')

def add_pos_tags():
    global POS_sents
    f = open('annotated_sentences_CLEANED_POS_STANFORD/PHYS/sentences.txt','r')
    lines = list(f.readlines())
    POS_sents = lines



def run():
    global word_count
    
    paras, sents = create_dataset()
    
    #write_sentences(paras)  
    
    X = np.array(get_features(paras))
    Y = np.array(get_ys(paras))
    
    feature_names = get_feature_names()

    for c, i in enumerate(feature_names):
        X_ = []
        for item in X:
            X_.append(item[c])
        X_ = np.array(X_)
        Y_ = np.array(Y).astype(np.float)
        print i
        print scipy.stats.pearsonr(X_, Y_)
    
    print ''
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    print "Features sorted by their RF score:"
    for n, p in sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), 
                 reverse=True):
        print p
        print n
      
    print ''  
    scores = defaultdict(list)
    rf = RandomForestRegressor()
    
    #crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(len(X), 10, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[feature_names[i]].append((acc-shuff_acc)/acc)
    
    print "Features sorted by their score:"
    for n, p in sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True):
        print p
        print n
        
add_pos_tags()
# load_liwc('affect')
# load_liwc('cogmech')
# load_liwc('percept')
# load_liwc('tentative')
run()

