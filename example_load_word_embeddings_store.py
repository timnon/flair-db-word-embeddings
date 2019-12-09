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
