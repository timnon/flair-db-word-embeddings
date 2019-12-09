import flair
from flair.embeddings import WordEmbeddings
import sqlite3
import torch
import re
import os
from tqdm import tqdm

class WordEmbeddingsStore():
    '''
    class to simulate a WordEmbeddings class from flair
    '''
    def __init__(self,embedding,verbose=True):
        # some non-used parameter to allow print
        self._modules = dict()
        self.items = ''

        # get db filename from embedding name
        self.name = embedding.name
        store_filename = WordEmbeddingsStore._get_store_filename(embedding)
        if verbose:
            print('store filename:',store_filename)

        # if embedding database already exists
        if os.path.isfile(store_filename):
            self.db = sqlite3.connect(store_filename)
            cursor = self.db.cursor()
            cursor.execute('SELECT * FROM embedding LIMIT 1;')
            result = list(cursor)
            self.k = len(result[0])-1
            return

        # otherwise, push embedding to database
        self.db = sqlite3.connect(store_filename)
        pwe = embedding.precomputed_word_embeddings
        self.k = pwe.vector_size
        self.db.execute(f"DROP TABLE IF EXISTS embedding;")
        self.db.execute(f"CREATE TABLE embedding(word text,{','.join('v'+str(i)+' float' for i in range(self.k))});")
        vectors_it = ( [word]+pwe.get_vector(word).tolist() for word in pwe.vocab.keys() )
        if verbose:
            print('load vectors to store')
        self.db.executemany(f"INSERT INTO embedding(word,{','.join('v'+str(i) for i in range(self.k))}) \
        values ({','.join(['?']*(1+self.k))})", tqdm(vectors_it))
        self.db.execute(f"DROP INDEX IF EXISTS embedding_index;")
        self.db.execute(f"CREATE INDEX embedding_index ON embedding(word);")
        self.db.commit()

    def _get_vector(self,word='house'):
        cursor = self.db.cursor()
        cursor.execute(f"SELECT * FROM embedding WHERE word='{word}';")
        result = list(cursor)
        if not result:
            return torch.tensor([0.0]*self.k)
        return result[0][1:]

    def embed(self,sentences):
        for sentence in sentences:
            for token in sentence:
                t = torch.tensor(self._get_vector(word=token.text.lower()))
                token.set_embedding(self.name,t)

    @staticmethod
    def _get_store_filename(embedding):
        '''
        get the filename of the store
        '''
        embedding_filename = re.findall('.flair(/.*)',embedding.name)[0]
        store_filename = str(flair.cache_root)+embedding_filename+'.sqlite'
        return store_filename

    @staticmethod
    def create_stores(model):
        '''
        creates database versions of all word embeddings in the model and
        deletes the original vectors to save memory
        '''
        for embedding in model.embeddings.embeddings:
            if type(embedding) == WordEmbeddings:
                WordEmbeddingsStore(embedding)
                del embedding.precomputed_word_embeddings

    @staticmethod
    def load_stores(model):
        '''
        loads the db versions of all word embeddings in the model
        '''
        for i,embedding in enumerate(model.embeddings.embeddings):
            if type(embedding) == WordEmbeddings:
                model.embeddings.embeddings[i] = WordEmbeddingsStore(embedding)

    @staticmethod
    def delete_stores(model):
        '''
        deletes the db versions of all word embeddings
        '''
        for embedding in model.embeddings.embeddings:
            store_filename = WordEmbeddingsStore._get_store_filename(embedding)
            if os.path.isfile(store_filename):
                print('delete store:',store_filename)
                os.remove(store_filename)
