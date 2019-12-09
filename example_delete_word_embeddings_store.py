import pickle
import time
from flair.models import SequenceTagger
from flair.data import Sentence
from WordEmbeddingsStore import WordEmbeddingsStore


tagger = SequenceTagger.load('multi-ner-fast')
WordEmbeddingsStore.delete_stores(tagger)
