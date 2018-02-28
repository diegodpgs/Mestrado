import MWESystem
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora


class Topics:

	def __init__(self,databasefile,path):
		self.MWES = MWESystem(databasefile,path)
		self.MWES.setup()
		self.docs = self.joinDocs()
		self.lemma = WordNetLemmatizer()
		self.ldamodel = None
		self.dictionary = None

	def joinDocs(self):
		d = {}


		for expression,data in self.MWES.mwe_windows:
			d[expression] = []
			for i in data:
				d[expression].append(i[2]) #[(__target__,__sentence_location__,window_sentence,file_path])...]
		return d

	
	def clean(self,doc):
	    
	    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	    normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
	    
	    return normalized


	def lda(self,expression,topics)
		doc_clean = [self.clean(doc).split() for doc in self.docs[expression]] 
		self.dictionary = corpora.Dictionary(doc_clean)
		doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in doc_clean]
		Lda = gensim.models.ldamodel.LdaModel
		self.ldamodel = Lda(doc_term_matrix, num_topics=topics, id2word = dictionary, passes=50)

	def lda_doc(self,doc,topics):
		r = (ldamodel.get_document_topics(self,dictionary.doc2bow(clean(doc).split()),per_word_topics=True))
		for j in r[0][0:topics]:
			print dictionary[j[0]],j[1]



if "__main__":
	t = Topics('cook_mwe.txt',os.getcwd()+'/Texts')