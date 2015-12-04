import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import f1_score, precision_score, recall_score,\
    accuracy_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm.classes import LinearSVC
from nltk.tokenize import word_tokenize
import re
import scipy.stats
from scipy.stats import pearsonr
import copy
import nltk
from nltk.corpus import stopwords
    
'''
Results (3-fold cross validation, ran once, Logistic Regression)

Paragraph Content (definition, summary) features only

Avg Acc      0.646453373016      
Avg Prec      0.772470811686
Avg Recall      0.311655575721
Avg F1      0.440926640927

Page Structure features only

Avg Acc      0.662409060847      
Avg Prec      0.65491274115
Avg Recall      0.640855011091
Avg F1      0.631623639086

Count Len Features

Avg Acc      0.628472222222      
Avg Prec      0.676790878295
Avg Recall      0.472978423069

Equation Features

Avg Acc      0.614996693122      
Avg Prec      0.556057831058
Avg Recall      0.757007461182
Avg F1      0.639755094707


Sentiment features

Avg Acc      0.599413029101      
Avg Prec      0.686473429952
Avg Recall      0.351986287558
Avg F1      0.404377026401

Example features

Avg Acc      0.557663690476      
Avg Prec      0.604062604063
Avg Recall      0.138435168381
Avg F1      0.214954278813

vocab features

Avg Acc      0.591559193122      
Avg Prec      0.693650793651
Avg Recall      0.17332123412
Avg F1      0.277302577515

para, page (page num only), equation, sentiment(remark, suggest, ! only), num sents, first occurence of vocab word, POS (JJ, CD)

Avg Acc      0.752087681375      
Avg Prec      0.754273504274
Avg Recall      0.698729582577
Avg F1      0.715064102564


'''


liwc = {}

number_words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", 
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion", "trillion"]

line_ct = 0
POS_sents = None

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
            'prev_para': copy.deepcopy(prev_para),
            'first_occur': [],
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
        
    non_vocab_occur = {}
    
    current_para = None
    
    first_page = int(lines[2].strip().split('\t')[0])
    
    for line in lines[2:]:
        page, head, c1, c2, c3, c4, sent = line.strip().split('\t')
                
        
        head_val = head.strip().split(' ')
        if head_val[0] == 'P' and not (len(head_val) == 2 and head_val[1] == 'SM'):
            if current_para != None:
                list_paras.append(current_para)
                current_para = create_para(head_val, page, c1, c2,c3, c4, sent, first_page, vocab_occur, current_para, non_vocab_occur)
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
    #val.append(para['EX'])
#     
#     # Number of Figures in the page
    #val.append(para['FIG'])
#     
#     # Number of Checkpoints in the page
    #val.append(para['HT'])
#     
#     # Number of H1 headers in the page (start of subchapter)
    #val.append(para['H1'])
#     
#     # Number of tables in the page
    #val.append(para['TBL'])
#     
#     # Number of procedures in the page
    #val.append(para['PR'])
#     
#     # Number of brown boxes in the page (side box?)
    #val.append(para['BR'])
#     
#     # Number of footnotes in the page
    #val.append(para['FN'])
    
    # page in the chapter
    val.append(para['page_num'])
    
#     if para['prev_para'] == None:
#         val.append(1)
#     else:
#         val.append(0)
    

def count_len_features(val, para):
    # Number of sentences in the para
    val.append(len(para['sents']))
    
    # Avg number of words per sentence
    #val.append(np.mean(para['word_ct']))

    # Avg length of sentence
    count = []
    for sent in para['sents']:
        count.append(len(sent))
    #val.append(np.mean(count))
    
    # total length of sentence
    count = 0
    for sent in para['sents']:
        count += len(sent)
    #val.append(count)
    
    # Avg number of chars per word
    count = []
    for sent in para['sents']:
        for word in word_tokenize(sent):
            count.append(len(word))
    #val.append(np.mean(count))

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
   # val.append(float(count)/float(len(para['sents'])))

    
def vocab_words_features(val, para):
    
    sent_text = ' '.join(para['sents'])
    
    # number of vocab words per sentence in the paragraph
    count = 0
    for sent, w in zip(para['sents'], para['word_ct']):
        for vocab in para['vocab'].keys():
            if re.search(r'\b%s\b' % vocab, sent.lower()):
                count += 1.0/float(len(vocab))
    #val.append(count)
    
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
    #val.append(np.mean(count) if count != [] else 0.0)
    
    
    
    # Max Num pages since last occurence
    count = 0
    for v in para['vocab']:
        if re.search(r'\b%s\b' % v, sent_text.lower()) and para['vocab'][v]['instance'] > 0:
            x = para['page'] - para['vocab'][v]['last_occur']
            if x > count:
                count = x
    #val.append(count)
               
    # Max Num pages since first occurence
    count = 0
    for v in para['vocab']:
        if re.search(r'\b%s\b' % v, sent_text.lower()) and para['vocab'][v]['instance'] > 0:
            x = para['page'] - para['vocab'][v]['first_occur']
            if x > count:
                count = x
    #val.append(count)
    
    
def _liwc_feature(feature, para):
    count = 0
    for item in liwc[feature]:
        for sent, w in zip(para['sents'], para['word_ct']):
            count += float(len(re.findall(item, sent.lower())))
    return count

def sentiment_of_words(val, para):
    
    # Occurrence of LIWC affect words
    #val.append(_liwc_feature('affect', para))
# # #     
# # #     # Occurrence of LIWC cogmech words
    #val.append(_liwc_feature('cogmech', para))
# # #     
# # #     # Occurrence of LIWC tentative words
    #val.append(_liwc_feature('tentative', para))
# # # 
# # #     # Occurrence of LIWC percept words
    #val.append(_liwc_feature('percept', para))
    
#     # Asking reader to remember something
#     count = 0
#     for sent in para['sents']:
#         sent = sent.lower()
#         count += sent.count('remember')
#         count += sent.count('recall')
#     val.append(count)
#     
#     count = 0
#     for sent in para['sents']:
#         sent = sent.lower()
#         count += sent.count('next page')
#         count += sent.count('next section')
#         count += sent.count('next chapter')
#     val.append(count)

    count = 0
    for sent in para['sents']:
        sent = sent.lower()
        count += sent.count('remarkable')
        count += 1 if re.search(r'surprising\b', sent) else 0
    val.append(count)
    
    
    count = 0
    for sent in para['sents']:
        sent = sent.lower()
        count += sent.count('suggest')
        count += sent.count('lead us to conclude')
        count += sent.count('leads us to conclude')
        count += sent.count('we can conclude')
        count += sent.count('mention')
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
#     count = 0
#     for sent in para['sents']:
#         count += sent.count('?')
#     val.append(count)
    
def example_words(val, para):
    
    sent_text = ' '.join(para['sents'])
    
    # occurrence of example phrases
#     if "examples" in sent_text:
#         val.append(1)
#     else:
#         val.append(0)
#         
#     if 'for example' in sent_text.lower() or 'for instance' in sent_text.lower():
#         val.append(1)
#     else:
#         val.append(0)

    #yous
    count = 0
    for sent in para['sents']:
        sent = sent.lower()
        count += sent.count('you ')
        count += sent.count('your ')
    #val.append(count)
    
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
            if pos in ['NN', 'NNS']:
                count += 1
            num += 1
    #val.append(float(count)/float(num))
    
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
    #val.append(np.mean(count))
    
    val.append(float(len(para['first_occur']))/float(np.sum(para['word_ct'])))


def get_features(paras):
    X = []
    for para in paras:
        val = []
        para_content_features(val, para)
        page_structure_features(val, para)
        equation_features(val, para)
        
        
        sentiment_of_words(val, para)
        vocab_words_features(val, para)
        count_len_features(val, para)
        
        pos_features(val, para)
        
        non_vocab_word_features(val, para)
        
        example_words(val, para)
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

def add_pos_tags():
    global POS_sents
    f = open('annotated_sentences_CLEANED_POS_STANFORD/PHYS/sentences.txt','r')
    lines = list(f.readlines())
    POS_sents = lines



def run():
    paras, sents = create_dataset()
    
    X = np.array(get_features(paras))
    Y = np.array(get_ys(paras))
    
    skf = StratifiedKFold(Y, n_folds=3)
    
    f = open('results/correct.txt','w')
    f2 = open('results/wrong.txt','w')
    
    accs = []
    precs = []
    recs = []
    f1s = []
    
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        clf = LogisticRegression()
        
        clf.fit(X_train, y_train)
        print clf.coef_
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        
        print 'Acc \t %s' % acc
        print 'Prec \t %s' % prec
        print 'Recall \t %s' % rec
        print 'F1 \t %s' % f1
        
        for (index,test),(y_t, y_p) in zip(zip(test_index, X_test), zip(y_test, y_pred)):
            if y_t == y_p:
                if paras[index]['prev_para']:
                    f.write('%s\n' % paras[index]['prev_para']['sents'])
                f.write('%s\n' % sents[index])
                f.write('%s - %s\n' % (y_t, test))
            else:
                if paras[index]['prev_para']:
                    f2.write('%s\n' % paras[index]['prev_para']['sents'])
                f2.write('%s\n' % sents[index])
                f2.write('%s - %s\n' % (y_t, test))
        
    print 'Avg Acc \t %s \t ' % np.mean(accs)
    print 'Avg Prec \t %s' % np.mean(precs)
    print 'Avg Recall \t %s' % np.mean(recs)
    print 'Avg F1 \t %s' % np.mean(f1s)
    

        
add_pos_tags()
load_liwc('affect')
load_liwc('cogmech')
load_liwc('percept')
load_liwc('tentative')
run()

