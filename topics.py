import MWESystem
import os
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora


class Topics:

	def __init__(self,databasefile,path):
		self.MWES = MWESystem.MWESystem(databasefile,path)
		self.MWES.setup()
		self.docs = self.joinDocs()
		self.lemma = WordNetLemmatizer()
		self.ldamodel = None
		self.dictionary = None
		self.punctuations = exclude = set(string.punctuation)
		self.doc_term_matrix = None
		self.docs_clean = None

	def joinDocs(self):
		d = {}

		for expression,data in self.MWES.mwe_windows.iteritems():
			d[expression] = []
			for i in data:
				d[expression].append(" ".join(i[2])) #[(__target__,__sente	nce_location__,window_sentence,file_path])...]
		return d

	
	def clean(self,doc):
	     
	    stop_free = " ".join([i for i in doc.lower().split() if i not in self.MWES.stopwords])
	    punc_free = ''.join(ch for ch in stop_free if ch not in self.punctuations)
	    normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
	    
	    return normalized


	def lda(self,expression,topics=3,words=3):
		self.doc_clean = [self.clean(doc).split() for doc in self.docs[expression]] 
		self.dictionary = corpora.Dictionary(self.doc_clean)
		self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in self.doc_clean]
		Lda = gensim.models.ldamodel.LdaModel
		self.ldamodel = Lda(self.doc_term_matrix, num_topics=topics, id2word = self.dictionary, passes=50)
		print self.ldamodel.print_topics(num_topics=topics)

	def lda_doc(self,doc,topics=3):
		return  self.ldamodel.get_document_topics(doc)


if "__main__":
	t = Topics('cook_mwe.txt',os.getcwd()+'/Texts')
	t.lda('blow_whistle')
	for i in xrange(len(t.doc_term_matrix)):
		print t.doc_clean[i]
		print t.lda_doc(t.doc_term_matrix[i])
		print '*'*40


# TODO: conforme a saida

# ****************************************
# reach the ball or try to do so england whistle blow league play allow

# testar com N tamanhos de topicos e de palavras
# mostrar qual texto esta sendo utilizado
# [(0, 0.017971505609338589), (1, 0.96420589942684309), (2, 0.017822594963818279)]
# mostrar os topicos juntamente com a pontuacao




