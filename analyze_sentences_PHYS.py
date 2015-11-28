#-*- coding: utf-8 -*-
from analyze_util import *
import unicodedata
from nltk.tokenize import sent_tokenize
from gensim.test.test_doc2vec import sentences


def write_stuff(page, source, frags, annots, heads, sentences, done=True):
    file2 = open('annotated_sentences/PHYS/%s.txt' % source, 'a')

    sentences = re.sub('—', '-', sentences)
    sentences = re.sub('', ' ', sentences)
    sentences = re.sub('•', '- ', sentences)
    sentences = re.sub('✔', '<CHECK>', sentences)
    sentences = re.sub('❶', '1) ', sentences)
    sentences = re.sub('❷', '2) ', sentences)
    sentences = re.sub('❸', '3) ', sentences)
    sentences = re.sub('“', '"', sentences)
    sentences = re.sub('”', '"', sentences)
    sentences = re.sub('’', '\'', sentences)
    
    sentences = unicode(sentences, 'utf-8')
    sentences = unicodedata.normalize('NFKD', sentences).encode('ascii','ignore')
            
    sents = sent_tokenize(sentences)
    print sents

    if done:
        for s in sents:
            
            if 'MAZU0' in s or 'Mazur' in s:
                continue
    
            max_c = 0
            headall = ''
            for frag, (annot, head) in zip(frags, zip(annots, heads)):
                if frag in s or s in frag:
                    if int(annot) > max_c:
                        #print count[1]
                        max_c = int(annot)
                else:
                    ss1 = commonOverlapIndexOf(s, frag)
                    ss2 = commonOverlapIndexOf(frag, s)
                    if ss1 > 10 or ss2 > 10:
                        if int(annot) > max_c:
                            #print count[1]
                            max_c = int(annot)
                headall = head
            #print max_c
            print s
            file2.write('%s\t%s\t%s\t%s\n' % (max_c, page, headall, s))
    else:
        
        for s in sents[:-1]:
            
            if 'MAZU0' in s or 'Mazur' in s:
                continue
    
            max_c = 0
            headall = ''
            for frag, (annot, head) in zip(frags, zip(annots, heads)):
                if frag in s or s in frag:
                    if int(annot) > max_c:
                        #print count[1]
                        max_c = int(annot)
                else:
                    ss1 = commonOverlapIndexOf(s, frag)
                    ss2 = commonOverlapIndexOf(frag, s)
                    if ss1 > 10 or ss2 > 10:
                        if int(annot) > max_c:
                            #print count[1]
                            max_c = int(annot)
                headall = head
            #print max_c
            print s
            file2.write('%s\t%s\t%s\t%s\n' % (max_c, page, headall, s))
            

        return [frags[-1]], [annots[-1]], [heads[-1]], sents[-1]


for source in PHYS_sources:
    
    print source
    
    frags = []
    annots = []
    heads = []
    
    file2 = open('annotated_sentences/PHYS/%s.txt' % source, 'w')
    file2.close()
    
    sentences = ''
    
    for page in range(1,160):
        try:
            file = open('annotated_confusion_results/%s/%s.txt' % (source, page))
        except:
            continue
        
        
        
        lines = file.readlines()
        if len(lines) < 15:
            continue
        
        figure_mode = False
        table_mode = False
        prev_annot = None
        for line in lines:
            if line.strip() == '':
                continue
            
            if line.strip() == '--TABLES--':
                table_mode = True
                continue
            
            if line.strip() == '--FIGURES--':
                if sentences != '':
                    frags, annots, heads, sentences = write_stuff(page, source, frags, annots, heads, sentences, done=False)
                figure_mode = True
                continue
            
            if table_mode:
                vals = line.strip().split('\t')
                write_stuff(page, source, [vals[1]], [vals[0]], ['TBL'], vals[1])
                continue
            
            if figure_mode:
                vals = line.strip().split('\t')
                write_stuff(page, source, [vals[1]], [vals[0]], ['FIG'], vals[1])
                continue
                
            vals = line.strip().split('\t')
            
            if len(vals) != 3:
                continue
            
            if vals[1] == 'H1' or (prev_annot != None and prev_annot != vals[1]):
                write_stuff(page, source, frags, annots, heads, sentences)
                frags = []
                annots = []
                heads = []

                sentences = ''
            
            frags.append(vals[2])
            annots.append(vals[0])
            heads.append(vals[1])
            x = vals[2]
            if x.endswith('-'):
                x = x[:-1]
            else:
                x = x.strip() + ' ' 
            sentences += x
            prev_annot = vals[1]
            
            
    if sentences != '':
        write_stuff(page, source, frags, annots, heads, sentences)
        frags = []
        annots = []
        heads = []
    
        sentences = ''
        
        
            
            