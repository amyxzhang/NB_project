import os
import psycopg2
import re
from analyze_util import *

conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost'")

cur = conn.cursor()

def extract_tables_figures(page_text, figures, tables, del_entries):
    i = 0
    while i < len(page_text):
        if re.match('^Figure [0-9]{1,2}.[0-9]{1,2}[a-z]?[^,.:a-zA-Z)]+[A-Z]', page_text[i]):
            fig = []
            fig.append(page_text[i])
            del_entries.append(i)
            x = i +1
            while x != len(page_text) and '__s__' in page_text[x] and not page_text[x].startswith('Figure ') and not page_text[x].startswith('Table '):
                fig.append(page_text[x])
                del_entries.append(x)
                x += 1
            i = x
            figures.append(fig)
        elif re.match('^Table [0-9]{1,2}.[0-9]{1,2}[a-z]?[^,.:a-zA-Z)]+[A-Z]', page_text[i]):
            table = []
            table.append(page_text[i])
            del_entries.append(i)
            x = i +1
            while x != len(page_text) and '__s__' in page_text[x] and not page_text[x].startswith('Figure ') and not page_text[x].startswith('Table '):
                table.append(page_text[x])
                del_entries.append(x)
                x += 1
            i = x
            tables.append(table)
        else:
            i += 1

def _write_col(file, counts, page_text):
    cur_pos = 0
    for c, (i, j) in enumerate(zip(counts, page_text)):
        
        if 'MAZU0930_PRIN_' not in j and 'Mazur' not in j:
            val = j.split('___')
            s = get_size(j)
            if s == 48.0:
                file.write('\n%s\tH1\t%s' % (i, val[0]))
                cur_pos = c
                continue
            if s == 36.0 and get_color(j) == 'rgb(172, 33, 63)':
                file.write('\n%s\tH2\t%s' % (i, val[0]))
                cur_pos = c
                continue
            if s == 32.0:
                file.write('\n%s\tH3\t%s' % (i, val[0]))
                cur_pos = c
                continue
            
            if c != 0 and ((len(val[0]) < 19 and counts[c] == counts[cur_pos]) or len(val[0]) < 3):
                file.write('%s' % val[0])
            elif c !=0 and (cur_pos == c-1 and len(page_text[cur_pos].split('___')[0]) < 6 and counts[c] == counts[cur_pos]):
                file.write('%s' % val[0])
            else:
                if s == 36.0:
                    file.write('\n%s\tSM\t%s' % (i, val[0]))
                else:
                    file.write('\n%s\t\t%s' % (i, val[0]))
                cur_pos = c


def write_page(source, page, counts, page_text):
    file = open('annotated_confusion_results/%s/%s.txt' % (source, page), 'w')
    
    # figure out where the lines start to know whether it is a 1 col or 2 col page
    vals = {}
    for item in page_text:
        val = item.split('___')
        if len(val[0]) > 50:
            left = round(float(val[1].split('__')[1][0:-2]))
            if left not in vals:
                vals[left] = 0
            vals[left] += 1
    
    # this pdf page only has 1 column
    if vals.get(216.0, 0) > 2 or (vals.get(72.0, 0) > 20 and vals.get(324.0, 0) < 8):
        _write_col(file, counts, page_text)
        
        
    # this pdf page has two columns. Write out left before right column
    else:  
        left_col_text = []
        left_col_count = []
        right_col_text = []
        right_col_count = []
        for c, p in zip(counts, page_text):
            val = p.split('___')
            if float(val[1].split('__')[1][0:-2]) < 295:
                left_col_text.append(p)
                left_col_count.append(c)
            else:
                right_col_text.append(p)
                right_col_count.append(c)
             
        _write_col(file, left_col_count, left_col_text)
        _write_col(file, right_col_count, right_col_text)
         
    # write out figures
    file.write('\n--FIGURES--\n')
    for i, j in zip(fig_counts, figures):
        file.write(str(i) + '\t')
        found_period = False
        for x in j:
            text = x.split('___')[0]
            if len(text) > 30 or text.strip().endswith('.') or not found_period:
                found_period = text.strip().endswith('.')
                file.write(text)
            else:
                break
        file.write('\n')
    
    # write out tables
    file.write('--TABLES--\n')
    for i, j in zip(table_counts, tables):
        file.write(str(i) + '\t')
        text = j[0].split('___')[0]
        file.write(text)
        file.write('\n')   
    
    
    

for source_num, source in enumerate(sources):
    
    initialize_paths(source)
    
    for page in range(1,110):
        
        page_text = get_page_text(cur, source, page)
        if page != 1:
            page_text = page_text[2:]
        
        figures = []
        tables = []
        del_entries = []       
        
        extract_tables_figures(page_text, figures, tables, del_entries)
                
        page_text = [i for j, i in enumerate(page_text) if j not in del_entries]
            
        counts = [0] * len(page_text)
        fig_counts = [0] * len(figures)
        table_counts = [0] * len(tables)
        
        if len(page_text) > 0:
    
            cur.execute("""
                        SELECT      E.body, L.id, L.ensemble_id
                        FROM        tbl_pdf_excerpts as E, base_location as L
                        WHERE       L.id = E.location_id AND E.source_id = %s AND E.page = %s;
                        """ % (source, page))
    
            rows = cur.fetchall()
            
            file_c = open('annotated_confused_comments/%s/%s.txt' % (source, page), 'w')
            file_d = open('annotated_comments/%s/%s.txt' % (source, page), 'w')
            
            
            for row in rows:
                location_id = row[1]
                ensemble_id = row[2]
                confused = analyze_comment_info(file_c, file_d, location_id, ensemble_id, source, page)

                if not confused:
                    continue
                
                text = row[0]
                text = text.split('\n')
                text = [t for t in text if t.strip() != '']
                
                positions = []
                fig_place = {}
                tab_place = {}
                for t in text:
                    indexes = [i for i,x in enumerate(page_text) if x == t]
                    if not indexes:
                        for k, fig in enumerate(figures):
                            fig_indexes = [i for i,x in enumerate(fig) if x == t]
                            if fig_indexes:
                                fig_place[k] = 1
                        for k, tab in enumerate(tables):
                            tab_indexes = [i for i,x in enumerate(tab) if x == t]
                            if tab_indexes:
                                tab_place[k] = 1
                    else:
                        counts[indexes[0]] += 1
                        
                for key in fig_place.keys():
                    fig_counts[key] += 1
                for key in tab_place.keys():
                    table_counts[key] += 1
                    
            
            write_page(source, page, counts, page_text)
 
        
        