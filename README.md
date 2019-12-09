# flair inference with low memory consumption

flair models eat a lot of memory to hold word embeddings, e.g. the model `multi-ner-fast` requires a total of 5gig memory for a single prediction (as measured by mprof). This is unnecessary at inference stage, since the lookup of word vectors can you externalized to some database, e.g. sqlite. This pushes the memory requirement down to only 700mb. The running time goes up on the other hand, but this is relatively insignificant for larger inference jobs. Follow this example to test:

Create db versions of the word embeddings, free memory, and write the smaller model to `tmp.pickle`:

```
from WordEmbeddingsStore import WordEmbeddingsStore
from flair.models import SequenceTagger
import pickle

tagger = SequenceTagger.load('multi-ner-fast')
WordEmbeddingsStore.create_stores(tagger)
pickle.dump(tagger,open('multi-ner-fast-headless.pickle','wb'))

```

Load the model and the db versions again and do inference with 700mb of memory:

```
import pickle
import time
from flair.data import Sentence
from WordEmbeddingsStore import WordEmbeddingsStore

tagger = pickle.load(open('multi-ner-fast-headless.pickle','rb'))
WordEmbeddingsStore.load_stores(tagger)

text = '''Schade um den Ameisenbären. Lukas Bärfuss veröffentlicht Erzählungen aus zwanzig Jahren. Darin geht es immer wieder um das Scheitern beim Erzählen. Am 2. November wird Lukas Bärfuss in Darmstadt der Georg-Büchner-Preis überreicht. Da muss ein aktuelles Buch her, dachte sich der Wallstein-Verlag und bringt am heutigen Montag, in der Herbstvorschau noch nicht angekündigt, einen Band mit Erzählungen heraus. «Malinois» heisst er, nach einem belgischen Schäferhund, der in der zwölften von dreizehn Geschichten eine unglückliche Rolle spielt: Er wird von einem Lieferwagen angefahren.'''

sentence = Sentence(text)
t_start = time.time()
tagger.predict(sentence)
print('new processing time:',time.time()-t_start)
print(sentence.get_spans('ner'))
```

These scripts are very beta and not much tested, so use with caution! The goal is to make a PR on the original flair to include this functionality in the right way.
