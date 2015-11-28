import os
import psycopg2
import re
from analyze_util import *
import unicodedata
from nltk.tokenize import sent_tokenize

conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost'")

cur = conn.cursor()


count_total = 0
sources = [
            #4848
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


for source in sources:

    initialize_paths(source)

    text_paras = []
    text_counts = []
    
    curr_counts = []
    
    cur_text = ''
    
    for page in range(1,60):
        
        page_text = get_page_text(cur, source, page)
        
        if len(page_text) == 0:
            break
            
        counts = [0] * len(page_text)

        cur.execute("""
                    SELECT      E.body, L.id, L.ensemble_id
                    FROM        tbl_pdf_excerpts as E, base_location as L
                    WHERE       L.id = E.location_id AND E.source_id = %s AND E.page = %s;
                    """ % (source, page))

        rows = cur.fetchall()
        
        file_c = open('annotated_confused_comments/%s/%s.txt' % (source, page), 'w')
        file_d = open('annotated_comments/%s/%s.txt' % (source, page), 'w')
        
        # for each confused comment, find where it takes place in the body text and increment the count
        for row in rows:
            location_id = row[1]
            ensemble_id = row[2]
            confused = analyze_comment_info(cur, file_c, file_d, location_id, ensemble_id, source, page)

            if not confused:
                continue
            
            text = row[0]
            text = text.split('\n')
            text = [t for t in text if t.strip() != '']
            
            positions = []

            for t in text:
                indexes = [i for i,x in enumerate(page_text) if x == t]
                if indexes:
                    counts[indexes[0]] += 1
            

        for i, j in zip(counts, page_text):
            val = j.split('___')
            if len(val[0].strip()) < 5:
                continue
            if 'M22_MAZU0930_PRIN_Ch22_pp593-614.indd' not in val[0]:
                
                x = re.sub('!', '', val[0])
                
                s = get_size(j)
                
                if s == 48.0:
                    if cur_text != '':
                        text_paras.append(cur_text)
                        text_counts.append(curr_counts)
                        curr_counts = []
                    
                    x = unicode(x, 'utf-8')
                    x = unicodedata.normalize('NFKD', x).encode('ascii','ignore')
                    text_paras.append([x, 'H2'])
                    text_counts.append(i)
                    cur_text = ''
                
                elif s == 56.0:
                    if cur_text != '':
                        text_paras.append(cur_text)
                        text_counts.append(curr_counts)
                        curr_counts = []
                        
                    x = unicode(x, 'utf-8')
                    x = unicodedata.normalize('NFKD', x).encode('ascii','ignore')
                    text_paras.append([x, 'H1'])
                    text_counts.append(i)
                    cur_text = ''
                
                else:
                    cur_text += x
                    if i != 0:
                        x = unicode(x, 'utf-8')
                        x = unicodedata.normalize('NFKD', x).encode('ascii','ignore')
                        curr_counts.append((i, x))

    if cur_text != '':
        text_paras.append(cur_text)
        text_counts.append(curr_counts)
        cur_text = ''

      
    final_sents = []
    final_counts = []
    for i, tc in zip(text_paras, text_counts):
        
        if type(i) != list:
            
            i = unicode(i, 'utf-8')
            xx = unicodedata.normalize('NFKD', i).encode('ascii','ignore')
            
            xx = re.sub('#', ' ', xx)
            
            sents = sent_tokenize(xx)

            for s in sents:
                print s
                final_sents.append(s)
                max_c = 0
                for count in tc:
                    ccc = re.sub('#', ' ', count[1])
                    if ccc in s or s in ccc:
                        if count[0] > max_c:
                            #print count[1]
                            max_c = count[0]
                    else:
                        ss1 = commonOverlapIndexOf(s, ccc)
                        ss2 = commonOverlapIndexOf(ccc, s)
                        if ss1 > 10 or ss2 > 10:
                            if count[0] > max_c:
                                #print count[1]
                                max_c = count[0]
                #print max_c
                final_counts.append(max_c)
        else:
            final_sents.append(i)
            final_counts.append(tc)
    
    file = open('annotated_sentences/BIO/%s.txt' % (source), 'w')
    
    for i, c in zip(final_sents, final_counts):
        if type(i) == list:
            file.write('%s\t%s\t%s\n' % (c, i[1], i[0]))
        else:
            if len(i) > 1:
                x = re.sub('  ', ' ', i)
                if x[0] == ' ':
                    x = x[1:]
                file.write('%s\t\t%s\n' % (c, x))
                count_total += 1

print count_total