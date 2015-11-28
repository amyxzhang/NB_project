import os
import psycopg2
from __builtin__ import str

conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost'")

cur = conn.cursor()

doc = 15657

for i in range(1,130):
    
    print i
    
    cur.execute("""
                SELECT         BL.id, BL.x, BL.y, BL.w, BL.h
                FROM        base_comment as BC, base_location as BL, base_source as S
                WHERE        BL.id = BC.location_id AND S.id = BL.source_id
                            AND S.id = %s AND BL.page = %s AND BC.parent_id IS NULL
                """ % (doc, i))
    
    rows = cur.fetchall()
    
    json_str = 'coords = ['
    for row in rows:
        str = '{"id": %s, "x": %s, "y": %s, "w": %s, "h": %s},' % (row[0],row[1],row[2],row[3],row[4])
        print row
        json_str += str
    json_str += '];'
    print json_str
    
    
    f = '/Users/axz/workspace/nb_project/pdfs/%s.split/%s.%s.pdf' % (doc, doc, i)
    
    path = '/Users/axz/workspace/nb_project/pdfs/%s.split/html' % doc
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    os.chdir( path)
    os.system('/usr/local/bin/pdf2htmlEX %s' % f)
    
    file = open('/Users/axz/workspace/nb_project/pdfs/%s.split/html/%s.%s.html' % (doc, doc, i) ,'r')
    
    html = file.read()
    
    ff = html.split('</body>')
    str_2 = '<script>%s</script>' % json_str
    new_str = '<script src="/static/nbapp/jquery-1.11.3.min.js"></script><script src="/static/nbapp/extract2.js"></script></body>'
    new_file = '%s%s%s%s' % (ff[0], str_2, new_str, ff[1])
    
    file.close()
    file2 = open('/Users/axz/workspace/nb_project/pdfs/%s.split/html/%s.%s.html' % (doc, doc, i),'w')
    file2.write(new_file)
    