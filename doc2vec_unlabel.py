import gensim
import csv
import logging
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os
from pprint import pprint

paragraphs = []
labels = []
#split the words from the sentence
tokenizer = RegexpTokenizer(r'\w+')
#stop words, e.g. that, for, on ...
stopword_set = set(stopwords.words("english"))

#clean data
def nlpClean(data):
	new_data = []
	for d in data:
		new_str = d.lower()
		dlist = tokenizer.tokenize(new_str)
		dlist = list(set(dlist).difference(stopword_set))
		new_data.append(dlist)
	return new_data

class LabeledLineSentence(object):
	"""docstring for LabeledLineSentence"""
	def __init__(self, doc_list, labels_list):
		self.labels_list = labels_list
		self.doc_list = doc_list

	def __iter__(self):
		for idx, doc in enumerate(self.doc_list):
			#Create new instance of TaggedDocument(words, tags)
			# pprint("line:"+ str(doc))
			# yield gensim.models.doc2vec.TaggedDocument(doc, [self.labels_list[idx]])
			yield gensim.models.doc2vec.LabeledSentence(doc, [idx])

# def getVecs(model, corpus, size):
# 	vecs = [np.array(model.doc2vecs[z.])]

# =================================== start init ========================================

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# load the test file
with open("title.csv") as csvfile:
	data_reader = csv.reader(csvfile)
	for idx, row in enumerate(data_reader):
		paragraphs.append(row[0])
		labels.append(row[2])		
new_paragraphs = nlpClean(paragraphs)
pprint(new_paragraphs)

# p_set = set()
# for x in new_paragraphs:
# 	for y in x:
# 		p_set.add(y)

# pprint(len(p_set))

# Process the train data
it = LabeledLineSentence(new_paragraphs, labels)

#size = number of the features, min_count = neglecting infrequent words, alpha = learning rate
model = gensim.models.Doc2Vec(it, size = 50,min_count=1) #alpha=0.025, min_alpha = 0.025)
model.save('doc2vec_ulb.model')
pprint("success 1")
# model.build_vocab(it)

pprint("doc2vecs length:" + str(len(model.docvecs)) )

outid = file("product_ulb_bpr_id_vector50.txt", "w")

for x in xrange(len(model.docvecs)):
	outid.write( str(x) + "\t")
	for idx, lv in enumerate(model.docvecs[x]):
		outid.write(str(lv)+ " ")
	outid.write("\n")

outid.close()

# error_num = 0
# for x in xrange(len(labels)):
# 	infer_vector = model.infer_vector(paragraphs[x])
# 	similar_documents = model.docvecs.most_similar([infer_vector], topn = 10)

# 	if similar_documents[0][0] != labels[x]:
# 		error_num = error_num + 1

# pprint("error rate:" + str(error_num) + "/" + str(len(labels)) + " = " + str(float(error_num)/len(labels)))
# # pprint(model.corpus_count)
# 	# pprint(similar_documents)