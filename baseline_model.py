import os
import sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import metrics
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm.classes import LinearSVC

'''
        unigram - Linear            0 only
BIO    0.693324616049    0.751626171805
PHYS    0.741477817895    0.736753559878
MSE    0.770224491467    0.813902743052

'''


sources = [
#             #5452
#             15640,
#             19250,
#             15639,
#             15638,
#             19251,
#             19252,
#             19253,
#             15641,
#             15637,
#             19254,
#             #4463
#             8324,
#             13876,
#             13232,
#             11978,
#             #5427
#             15443,
#             15442,
#             15433,
#             15439,
#             15430,
#             #6037
#             19256,
#             19258,
#             19259,
#             19257,
#             19260,
#             19255,
#            #5462    MSE
#             15660,
#             15651,
#             15659,
#             15645,
#             15652,
#             15657,
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



path = 'annotated_confusion_results/%s'


total = []

for source in sources:
    # predict this source 
    print source
    
    train_set = []
    test_set = []
    
    corpus_train = []
    corpus_test = []
    
    y_train = []
    y_test = []
    vectorizer = TfidfVectorizer()
    
    for source2 in sources:
        if source2 != source:
            s_path = path % source2
            for file in os.listdir(s_path):
                
                f = open('%s/%s' % (s_path, file),'r')
                lines = f.readlines()
                
                for line in lines[1:]:
                    if 'FIGURES' in line:
                        break
                    vals = line.strip().split('\t')
                    text = vals[2]
                    corpus_train.append(text)
                    if int(vals[0]) == 0:
                        y_train.append('0')
                    else:
                        y_train.append('1')
        else:
            s_path = path % source2
            for file in os.listdir(s_path):
                
                f = open('%s/%s' % (s_path, file),'r')
                lines = f.readlines()
                
                for line in lines[1:]:
                    if 'FIGURES' in line:
                        break
                    vals = line.strip().split('\t')
                    text = vals[2]

                    corpus_test.append(text)
                    if int(vals[0]) == 0:
                        y_test.append('0')
                    else:
                        y_test.append('1')
    
    X_train = vectorizer.fit_transform(corpus_train)

    X_test = vectorizer.transform(corpus_test)
    
    clf = RandomForestClassifier(n_estimators=10)
    #clf = KNeighborsClassifier(n_neighbors=10)
    #clf = LinearSVC()
    
    clf.fit(X_train, y_train)
    
    print len(y_train)
    print len(y_test)
    
    pred = clf.predict(X_test)
    
    #pred = ['0']* len(y_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    total.append(score)
    
    n = 20
    
#     feature_names = vectorizer.get_feature_names()
#     coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
#     top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
#     for (coef_1, fn_1), (coef_2, fn_2) in top:
#         print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)                   
                    
    
print np.mean(total)
