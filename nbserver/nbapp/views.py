from django.shortcuts import render
from cgitb import text
from django.views.decorators.csrf import csrf_exempt
from django.http.response import HttpResponse
import psycopg2
import json


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


# Create your views here.

@csrf_exempt
def get_confused(request):
    id = request.GET['id']
    
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost'")
    cur = conn.cursor()
    
    cur.execute("""
    
    SELECT C.body, C.author_id
    FROM base_comment as C
    Where C.location_id = %s AND C.parent_id IS NULL;
    """ % (id))
    
    rows = cur.fetchall()
    
    confused = is_confused(rows[0][0].lower())
    
    dict = {'text': rows[0][0], 
            'confused': str(confused), 
            'id': rows[0][1]}
    
    return HttpResponse(json.dumps(dict))
    
    

@csrf_exempt
def set_text(request):
    
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost'")

    cur = conn.cursor()

    page = request.POST['page']
    id = request.POST['id']
    text = request.POST['text']
    x = request.POST['x']
    y = request.POST['y']
    w = request.POST['w']
    h = request.POST['h']
    
    val = page.split('/')[-1]
    val = val.split('.')
    source = val[0]
    page_num = val[1]
    
    
    cur.execute(
     """INSERT INTO tbl_pdf_excerpts (location_id, source_id, x, y, w, h, page, body)
         VALUES (%s, %s, %s, %s, %s, %s, %s, %s);""",
     (id, source, x, y, w, h, page_num, text))
    
    print page_num

    print text
    
    conn.commit()
    
    cur.close()
    conn.close()
    
    return HttpResponse()


@csrf_exempt
def set_text_page(request):
    
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost'")

    cur = conn.cursor()

    page = request.POST['page']
    text = request.POST['text']
    
    val = page.split('/')[-1]
    val = val.split('.')
    source = val[0]
    page_num = val[1]
    
    cur.execute(
     """INSERT INTO tbl_pdf_text (source_id, page, body)
         VALUES (%s, %s, %s);""",
     (source, page_num, text))

    print page_num
    print text
    
    conn.commit()
    
    cur.close()
    conn.close()
    
    return HttpResponse()
    