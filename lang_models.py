''' Python3 code '''


import nltk
from nltk.corpus import brown
from nltk import bigrams, ngrams, trigrams 
import re
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import math


TEST_CASES_FILE="test_cases.txt"



sents=[s for s in brown.sents()]		# the entire corpus - sentences as collection of words. The NLTK list types...
										# ...  are custom "lazy" types and could not be directly edited


# regex to remove special characters and numerals
for s_idx,sentence in enumerate(sents):
	for idx,word in enumerate(sentence):
		sents[s_idx][idx]=re.sub('[^a-zA-Z]+','',word)
	sents[s_idx]=[word.lower() for word in sents[s_idx] if len(word)>0]

sents_test=sents[40000:]
sents=sents[:40000]

print ("> Preprocessing Finished")


# GLOBAL VARIABLES
bi_model_counts={}
bi_model_counts_distinct={}			# number of unique bigrams beginning with some word
tri_model_counts={}
tri_model_counts_distinct={}		# number of unique trigrams beginning with some word tuple (w1,w2)

tot_unigram=None



def build_models(k=0,smoothing=None,plot=False):
	global bi_model_counts,bi_model_counts_distinct,tri_model_counts,tri_model_counts_distinct,tot_unigram

	unigram_collection=[]
	bigram_collection=[]
	trigram_collection=[]
	for sentence in sents:
		unigram_collection.extend([word for word in sentence])
		bigram_collection.extend(nltk.bigrams(sentence,pad_left=True,pad_right=True))
		trigram_collection.extend(nltk.trigrams(sentence,pad_left=True,pad_right=True))

	unigram_counts=Counter(unigram_collection)
	bigram_counts=Counter(bigram_collection)
	trigram_counts=Counter(trigram_collection)

	unigram_counts['START']=40000
	unigram_counts['\START']=40000
	# creating array with counts of tokens with r occurrences for good turing smoothing
	turing_bigram=np.zeros(bigram_counts.most_common(n=1)[0][1]+1)
	turing_trigram=np.zeros(trigram_counts.most_common(n=1)[0][1]+1)
	for token in bigram_counts:
		turing_bigram[bigram_counts[token]]+=1
	turing_bigram[0]=len(unigram_counts.keys())**2-len(bigram_counts.keys())
	
	for token in trigram_counts:
		turing_trigram[trigram_counts[token]]+=1
	turing_trigram[0]=len(unigram_counts.keys())**3-len(trigram_counts.keys())


	r_bigram=np.zeros(bigram_counts.most_common(n=1)[0][1]+1)
	for  idx,_ in enumerate(r_bigram[:-1]):
		# find the next non-zero element in bigram_counts
		next_r=turing_bigram.nonzero()[0][turing_bigram.nonzero()[0]>idx][0]
		r_bigram[idx]=next_r*(turing_bigram[next_r]*1./turing_bigram[idx])

	r_trigram=np.zeros(trigram_counts.most_common(n=1)[0][1]+1)
	for  idx,_ in enumerate(r_trigram[:-1]):
		# find the next non-zero element in bigram_counts
		next_r=turing_trigram.nonzero()[0][turing_trigram.nonzero()[0]>idx][0]
		r_trigram[idx]=next_r*(turing_trigram[next_r]*1./turing_trigram[idx])



	#r_bigram=(turing_bigram[1:]*np.array(range(1,turing_bigram.size)))/turing_bigram[:-1]
	n_bigram=np.sum(turing_bigram[1:]*np.array(range(1,turing_bigram.size)))
	#r_trigram=(turing_trigram[1:]*np.array(range(1,turing_trigram.size)))/turing_trigram[:-1]
	n_trigram=np.sum(turing_trigram[1:]*np.array(range(1,turing_trigram.size)))



	### plotting for Zipfs law verification #############################################
	if plot:
		n_plot=500
		zipf=np.log10(1./np.array(range(1,n_plot+1)))
		x_1=np.log10(range(1,n_plot+1))

		y_1=np.zeros(n_plot)
		for idx,(token,val) in enumerate(unigram_counts.most_common(n=n_plot)):
			if idx==n_plot:break
			y_1[idx]=val
		y_1=np.log10(y_1/y_1[0])
		l1,l2=plt.plot(x_1,y_1,x_1,zipf)
		plt.legend([l1,l2],['unigram counts','ideal zipf'])
		plt.savefig('unigram.png')


		plt.clf()
		y_1=np.zeros(n_plot)
		for idx,(token,val) in enumerate(bigram_counts.most_common(n=n_plot)):
			if idx==n_plot:break
			y_1[idx]=val
		y_1=np.log10(y_1/y_1[0])
		l1,l2=plt.plot(x_1,y_1,x_1,zipf)
		plt.legend([l1,l2],['bigram counts','ideal zipf'])
		plt.savefig('bigram.png')


		plt.clf()
		y_1=np.zeros(n_plot)
		for idx,(token,val) in enumerate(trigram_counts.most_common(n=n_plot)):
			if idx==n_plot:break
			y_1[idx]=val
		y_1=np.log10(y_1/y_1[0])
		l1,l2=plt.plot(x_1,y_1,x_1,zipf)
		plt.legend([l1,l2],['trigram counts','ideal zipf'])
		plt.savefig('trigram.png')

		print("TOP 10 UNIGRAMS: ")
		for token in unigram_counts.most_common(n=10):
			print (token)

		print("TOP 10 BIGRAMS: ")
		for token in bigram_counts.most_common(n=10):
			print (token)


		print("TOP 10 TRIGRAMS: ")
		for token in trigram_counts.most_common(n=10):
			print (token)
	################################################################################




	# creating the uni-, bi- and tri- gram models
	unigram_model=unigram_counts
	bigram_model=bigram_counts
	trigram_model=trigram_counts


	#THE UNGRAM MODEL
	tot_unigram=len(unigram_counts)
	for token in unigram_model:
		unigram_model[token]/=(tot_unigram*1.)

	# THE BIGRAM MODEL

	for w1,w2 in bigram_model:
		if w1 not in bi_model_counts:
			bi_model_counts[w1]=bigram_model[(w1,w2)]
			bi_model_counts_distinct[w1]=0
		else:
			bi_model_counts[w1]+=bigram_model[(w1,w2)]
			bi_model_counts_distinct[w1]+=1
	for w1,w2 in bigram_model:
		if smoothing is None:
			bigram_model[(w1,w2)]/=(float(bi_model_counts[w1]))
		
		elif smoothing=='laplacian':
			bigram_model[(w1,w2)]+=k
			bigram_model[(w1,w2)]/=((float(bi_model_counts[w1]))+k*tot_unigram)#bi_model_counts_distinct[w1]
		
		elif smoothing=='good_turing':
			bigram_model[(w1,w2)]=r_bigram[bigram_counts[(w1,w2)]]/n_bigram


		else:
			print("Invalid value for 'smoothing'")
			exit()


	# THE TRIGRAM MODEL
	for w1,w2,w3 in trigram_model:
		if (w1,w2) not in tri_model_counts:
			tri_model_counts[(w1,w2)]=trigram_model[(w1,w2,w3)]
			tri_model_counts_distinct[(w1,w2)]=0
		else:
			tri_model_counts[(w1,w2)]+=trigram_model[(w1,w2,w3)]
			tri_model_counts_distinct[(w1,w2)]+=1
	for w1,w2,w3 in trigram_model:
		if smoothing is None:
			trigram_model[(w1,w2,w3)]/=(float(tri_model_counts[(w1,w2)]))
		
		elif smoothing =='laplacian':
			trigram_model[(w1,w2,w3)]+=k
			trigram_model[(w1,w2,w3)]/=((float(tri_model_counts[(w1,w2)]))+k*tot_unigram)#tri_model_counts_distinct[(w1,w2)]

		elif smoothing=='good_turing':
			trigram_model[(w1,w2,w3)]=r_trigram[trigram_counts[(w1,w2,w3)]]/n_trigram

		else:
			print("Invalid value for 'smoothing'")
			exit()


	if smoothing=='good_turing':
		bigram_model['DEFAULT_VALUE']=r_bigram[0]/n_bigram
		trigram_model['DEFAULT_VALUE']=r_trigram[0]/n_trigram

	print ("> Language Models Created")
	return (unigram_model,bigram_model,trigram_model)





def get_scores(model,sents,n,smoothing=None,k=1):
	'''
	model: a counter object
	sents: Expects a list of preprocessed strings
	smoothing: {None,'laplacian','good_turing'}
	k: parameter for laplacian smoothing
	'''

	scores=[]
	perpl=[]
	for sentence in sents:
		sentence=sentence.split()
		p=0.
		if n>=2:
			itr=ngrams(sentence,n)
		else:
			itr=sentence
		for token in itr:
			try:
				p+=math.log(model[token],2)
			except:
				if smoothing=='laplacian':
					if n==1:
						p+=math.log((1./tot_unigram),2)
					elif n==2:
						p+=math.log(k/(k*tot_unigram+bi_model_counts[token[0]]),2)
					else:
						p+=math.log(k/(k*tot_unigram*1.0+tri_model_counts[token[0],token[1]]),2)
				elif smoothing=='good_turing':
					p+=math.log(model['DEFAULT_VALUE'],2)
					
				else:
					p+=(-1*np.inf)
		scores.append(p)
		perpl.append(math.pow(2.,(-1./len(list(ngrams(sentence,n))))*p))

	return np.array(scores),np.array(perpl)


def bigram_interpol(unigram_model,bigram_model,lamb,sents):
	'''
	Creates the interpolation model and returns the same
	'''
	scores=[]
	perpl=[]
	for sentence in sents:
		sentence=sentence.split()
		p=0.	
		for w1,w2 in ngrams(sentence,2):
			if w2 in unigram_model and (w1,w2) in bigram_model:
				p+=math.log((lamb*bigram_model[(w1,w2)])+(1-lamb)*unigram_model[w2],2)
			elif w2 in unigram_model:
				p+=math.log((1-lamb)*unigram_model[w2],2)
			elif (w1,w2)in bigram_model:
				p+=math.log((lamb*bigram_model[(w1,w2)]),2)
			else:
				p+=(-1*np.inf)
		scores.append(p)
		perpl.append(math.pow(2.,(-1./len(list(ngrams(sentence,2))))*p))

	return np.array(scores),np.array(perpl)

def driver():
	np.set_printoptions(precision=3)		
	np.seterr(divide='ignore')
	

	TEST_CASES_FILE=input("Enter the name of test cases file: > ")


	sents=[]
	with open(TEST_CASES_FILE,"r") as input_file:
		for line in input_file:
			line=re.sub('[^a-zA-Z ]+','',line)
			sents.append(line)

	n=[1,2,3]

	models=build_models(plot=True)
	model_org=models
	print("\n\nNO SMOOTHING: ")
	for n_val in n:
		s,p=get_scores(models[n_val-1],sents,n_val)
		print("at n = ",n_val," score = ",s,"; perpl = ",p)

	## LAPLACIAN SMOOTHING ##
	print("\n\nUSING LAPLACIAN SMOOTHING: ")
	smoothing='laplacian'
	k=[0.0001,0.001,0.01,0.1,1.0]
	
	for k_val  in k:
		models=build_models(k_val,smoothing)
		for n_val in n:
			s,p=get_scores(models[n_val-1],sents,n_val,smoothing,k_val)
			print("at n = ",n_val,", k = ",k_val," score = ",s,"; perpl = ",p)


	## GOOD TURING SMOOTHING
	print ("\n\nUSING TURING SMOOTHING")
	smoothing='good_turing'
	models=build_models(smoothing=smoothing)
	for n_val in n:
		s,p=get_scores(models[n_val-1],sents,n_val,smoothing)
		print("at n = ",n_val," score = ",s,"; perpl = ",p)
	


	# INTERPOLATION:
	print("\n\nUSING INTERPOLATION:")
	lambda_val=[.2,.5,.8]	
	for lamb in lambda_val:
		s,p=bigram_interpol(model_org[0],model_org[1],lamb,sents)
		print("at n = 2, lambda = ",lamb," score = ",s,"; perpl = ",p)

driver()
	
	
