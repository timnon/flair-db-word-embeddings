# flair inference with low memory consumption

flair models eat a lot of memory to hold word embeddings, e.g. the model `multi-ner-fast` requires a total of 5gig memory for a single prediction (as measured by mprof). This is unnecessary at inference stage, since the lookup of word vectors can you externalized to some database, e.g. sqlite. This pushes the memory requirement down to only 700mb. The running time goes up on the other hand, but this is relatively insignificant for larger inference jobs. Follow this example to test:

Create db versions of the word embeddings, free memory, and write the smaller model to `tmp.pickle`:

```
from flair.models import SequenceTagger
import pickle
from dbwordembedding import *

tagger = SequenceTagger.load('multi-ner-fast')
create_db_word_embeddings(tagger)
pickle.dump(tagger,open('tmp.pickle','wb'))
```

Load the model and the db versions again and do inference with 700mb of memory:

```
import time
from flair.data import Sentence

tagger = pickle.load(open('tmp.pickle','rb'))
load_db_word_embeddings(tagger)

sentence = Sentence('Hier wohnt Hans Mustermann')
t_start = time.time()
tagger.predict(sentence)
print('new processing time:',time.time()-t_start)
print(sentence.get_spans('ner'))
```

These scripts are very beta and not much tested, so use with caution! The goal is to make a PR on the original flair to include this functionality in the right way.
