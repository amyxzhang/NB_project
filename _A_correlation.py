
from scipy.stats.stats import pearsonr

def run(val):
    print val
    file = open('annotated_sentences_CLEANED/PHYS_dedup/4_count/%s.txt' % val,'r')
    
    lines = file.readlines()
    
    vals1 = []
    vals2 = []
    vals3 = []
    vals4 = []
    
    vocab = lines[0]
    for line in lines[2:]:
        page, head, c1, c2, c3, c4, sent = line.strip().split('\t')
        
        if head == '':
            c1 = int(c1)
            c2 = int(c2)
            c3 = int(c3)
            c4 = int(c4)
            page = int(page)
            
            vals1.append(c1)
            vals2.append(c2)
            vals3.append(c3)
            vals4.append(c4)
        
    print pearsonr(vals1, vals2)
    print pearsonr(vals2, vals3)
    print pearsonr(vals3, vals4)
    print pearsonr(vals1, vals3)
    print pearsonr(vals1, vals4)
    print pearsonr(vals2, vals4)
    print pearsonr(vals3, vals4)
    print '----'
    
    
run(22)
run(23)
run(24)
run(30)
run(31)
run(32)
run(33)
run(34)
