import nltk

from analyze_util import *


for source in BIO_sources:
    f = open('annotated_sentences_CLEANED/BIO/%s.txt' % source,'r')
    f2 = open('annotated_sentences_CLEANED_POS/BIO/%s.txt' % source,'w')
    lines = f.readlines()
    for line in lines:
        vals = line.strip().split('\t')
        sent = vals[2]
        words = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(words)
        sent_tagged = []
        for tag in tags:
            sent_tagged.append('%s__%s' % (tag[0], tag[1]))
        new_vals = '%s\t%s\t%s\n' % (vals[0], vals[1], ' '.join(sent_tagged))
        f2.write(new_vals)