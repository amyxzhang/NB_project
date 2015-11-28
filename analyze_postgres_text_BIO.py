import os
import psycopg2
import re
from analyze_util import *

conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost'")

cur = conn.cursor()

def write_file(source, page, counts, page_text):
    # write the counts and the body text to file
    file = open('annotated_confusion_results/%s/%s.txt' % (source, page), 'w')

    for i, j in zip(counts, page_text):
        val = j.split('___')
        if len(val[0].strip()) < 5:
            continue
        if 'M22_MAZU0930_PRIN_Ch22_pp593-614.indd' not in val[0]:
            
            x = re.sub('!', '', val[0])
            
            s = get_size(j)
            
            if s == 48.0:
                file.write('%s\tH2\t%s\n' % (i, x))
            
            elif s == 56.0:
                file.write('%s\tH1\t%s\n' % (i, x))
            
            else:
                file.write('%s\t\t%s\n' % (i, x))
    
for source in sources:

    initialize_paths(source)
    
    for page in range(1,110):
        
        page_text = get_page_text(cur, source, page)
        
        counts = [0] * len(page_text)
        
        if len(page_text) > 0:
    
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
            
            write_file(source, page, counts, page_text)
