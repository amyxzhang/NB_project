import gensim
from analyze_util import *
import nltk
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec

sents = []

for source in BIO_sources:
    f = open('annotated_sentences_CLEANED_RAW/BIO/%s' % source, 'r')
    lines = f.readlines()
    for line in lines:
        
        line = re.sub("[^a-zA-Z]"," ", line).lower()
        
        ws = nltk.word_tokenize(line)
        
        stops = set(stopwords.words("english"))
        words = [w for w in ws if not w in stops]
        
        sents.append(ws)


model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#model = gensim.models.Word2Vec(sents, min_count=1)

print model.most_similar('network')


