from django.shortcuts import render
from cgitb import text
from django.views.decorators.csrf import csrf_exempt
from django.http.response import HttpResponse
import psycopg2



# Create your views here.

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
    