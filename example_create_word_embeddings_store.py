from WordEmbeddingsStore import WordEmbeddingsStore
from flair.models import SequenceTagger
import pickle

tagger = SequenceTagger.load('multi-ner-fast')
WordEmbeddingsStore.create_stores(tagger)
pickle.dump(tagger,open('multi-ner-fast-headless.pickle','wb'))
