import re
import os

BIO_sources = [
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

MSE_sources = [
           #5462
           15660,
           15651,
           15659,
           15645,
           15652,
           15657,
]

PHYS_sources = [
            #5452
            15640,
            19250,
            15639,
            15638,
            19251,
            19252,
            19253,
            15641,
            15637,
            19254,
            #4463
            8324,
            13876,
            13232,
            11978,
            #5427
            15443,
            15442,
            15433,
            15439,
            15430,
            #6037
            19256,
            19258,
            19259,
            19257,
            19260,
            19255,
]

def commonOverlapIndexOf(text1, text2):
    # Cache the text lengths to prevent multiple calls.
    text1_length = len(text1)
    text2_length = len(text2)
    # Eliminate the null case.
    if text1_length == 0 or text2_length == 0:
        return 0
    # Truncate the longer string.
    if text1_length > text2_length:
        text1 = text1[-text2_length:]
    elif text1_length < text2_length:
        text2 = text2[:text1_length]
    # Quick check for the worst case.
    if text1 == text2:
        return min(text1_length, text2_length)
    
    # Start by looking for a single character match
    # and increase length until no match is found.
    best = 0
    length = 1
    while True:
        pattern = text1[-length:]
        found = text2.find(pattern)
        if found == -1:
            return best
        length += found
        if text1[-length:] == text2[:length]:
            best = length
            length += 1

def get_page_text(cur, source, page):
    cur.execute("""
                    SELECT      E.body
                    FROM        tbl_pdf_text as E
                    WHERE       E.source_id = %s AND E.page = %s;
                    """ % (source, page))

    rows = cur.fetchall()
    
    page_text = ""
    
    for row in rows:
        page_text = row[0]
    
    page_text = page_text.split('\n')
    page_text = [p for p in page_text if p.split('___')[0].strip() != '']
    
    return page_text

def initialize_paths(source):
    directory = 'annotated_confused_comments/%s' % source 
    directory2 =  'annotated_comments/%s' % source 
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    
    path = 'annotated_discussion_results/%s/' % source
    path2 = 'annotated_confusion_results/%s/' % source
    
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path2):
        os.makedirs(path2)

def get_size(text):
    val = text.split('___')
             
    for k in val[1:]:
        if k.startswith('s__'):
            s = round(float(k.split('__')[1][0:-2]))
            return s
    return None

def get_color(text):
    val = text.split('___')
             
    for k in val[1:]:
        if k.startswith('c__'):
            s = k.split('__')[1]
            return s
    return None
    

def is_confused(text):
    t = text.lower()
    if '?' in t:
        return True
    
    if "i'm having trouble" in t:
        return True

    if "what about" in t:
        return True
    
    if "explanation as to why" in t:
        return True

    if "this section is rough" in t:
        return True
    
    if "not sure" in t:
        return True
    
    if "understand" in t:
        return True
    
    if "confusing" in t:
        return True
    
    if "what is the" in t:
        return True
    
    if "when can we" in t:
        return True
    
    if "correct me if I am wrong" in t:
        return True
    
    if "unclear" in t:
        return True
    
    if "are we going to" in t:
        return True
    
    if "i wonder" in t:
        return True

    return False

def analyze_comment_info(cur, file_c, file_d, location_id, ensemble_id, source, page):
    
    cur.execute("""
            SELECT      C.id, C.parent_id, C.body, M.admin
            FROM        base_comment as C, base_user as U, base_membership as M
            WHERE       U.id = C.author_id AND U.id = M.user_id AND M.ensemble_id = %s AND
                        C.location_id = %s;
            """ % (ensemble_id, location_id))
    
    rows = cur.fetchall()

    confused = False
    
    for row in rows:
        if row[1] == None:
            x = re.sub('\n', '', row[2])
            
            if is_confused(x):
                confused = True
                file_c.write('%s\t%s\t%s\n' % (row[0], row[3], x))
            else:
                file_d.write('%s\t%s\t%s\n' % (row[0], row[3], x))
                #print 'NC - %s' % x
        else:
            pass
           # print '\t%s\t %s' % (row[3], row[2])
        
    
    return confused


PHYS_chapters = {
            22: {15430: (0,22),
                 19250: (0,22),
                 19256: (0,22),
                 11978: (0,22)
                 },
            23: {15430: (23,46),
                 19250: (23,46),
                 19256: (23,46),
                 11978: (23,46)
                 },
            24: {15430: (47, -1),
                 19250: (47, -1),
                 19256: (47, -1),
                 11978: (47, -1)
                 },
            25: {15433: (0, 22),
                 19251: (0, 22),
                 19257: (0, 22)
                 },
            26: {15433: (23, -1),
                 19251: (23, -1),
                 19257: (23, -1)
                 },
            27: {19252: (0, 25),
                 19258: (0, 25),
                 },
            28: {19252: (26, 49),
                 19258: (26, 49),
                 },
            29: {19252: (50, -1),
                 19258: (50, -1),
                 },
            30: {13232: (0, 30),
                 15442: (0, 30),
                 19253: (0, 30),
                 19259: (0, 30)
                 },
            31: {13232: (31,61),
                 15442: (31,61),
                 19253: (31,61),
                 19259: (31,61)
                 },           
            32: {13232: (62, -1),
                 15442: (62, -1),
                 19253: (62, -1),
                 19259: (62, -1)
                 }, 
            33: {13876: (0, 33),
                 15439: (0, 33),
                 19254: (0, 33),
                 19260: (0, 33),
                 },
            34: {13876: (34, -1),
                 15439: (34, -1),
                 19254: (34, -1),
                 19260: (34, -1),
                 },
            15: {15641: (0, 26)
                 },
            16: {15641-16: (27, 59),
                 19255: (0, -1)
                 },
            17: {15641: (60, -1)
                 },
            1:  {15637: (0, 27),
                 8324: (0, 28)
                 },
            2:  {15637: (28, 52),
                 8324: (29, 53)
                 },
            3:  {15637: (53, -1),
                 8324: (54, -1)
                 },
            4:  {15638: (0, 26)
                 },
            5:  {15638: (27, 46)
                 },
            6:  {15638: (47, -1)
                 },    
            7:  {15639: (0, 28)
                 },
            8:  {15639: (29, 54)
                 },
            9:  {15639: (55, -1)
                 },     
            10:  {15640: (0, 28)
                 },
            11:  {15640: (29, 55)
                 },
            12:  {15640: (56, 82)
                 },    
            13:  {15640: (83, -1)
                 },       
            }