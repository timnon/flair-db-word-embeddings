import pickle
import time
from flair.models import SequenceTagger
from flair.data import Sentence
from dbwordembedding import *

tagger = pickle.load(open('tmp.pickle','rb'))
load_db_word_embeddings(tagger)

sentence = Sentence('Hier wohnt Hans Mustermann')
t_start = time.time()
tagger.predict(sentence)
print('new processing time:',time.time()-t_start)
print(sentence.get_spans('ner'))

# # to check original processing time
# sentence = Sentence('Hier wohnt Hans Mustermann')
# tagger = SequenceTagger.load('multi-ner-fast')
# t_start = time.time()
# tagger.predict(sentence)
# print('original processing time:',time.time()-t_start)
# print(sentence.get_spans('ner'))
