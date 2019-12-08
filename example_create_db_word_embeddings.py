from dbwordembedding import *
from flair.models import SequenceTagger
import pickle

tagger = SequenceTagger.load('multi-ner-fast')
create_db_word_embeddings(tagger)
pickle.dump(tagger,open('tmp.pickle','wb'))
