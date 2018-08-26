#NaiveBayes.py
from h2o.estimators import H2ONaiveBayesEstimator
import nltk 

#from BayesData import BayesData 
#from Tokenizer import Tokenizer 
from StopDicts import StopWords, StopPunctuation

import numpy as np 
import pandas as pd
from collections import namedtuple 
import operator
from tqdm import tqdm 
############################################
# Set up datasets for bayes classification # 
############################################
BayesData = namedtuple("BayesData", ["RussianData",
									 "HillaryData",
									 "TrumpData",
									 "NoneData"])

bayesData = BayesData(None, None, None, None)

# russian dataset comes with the columns:
# user_id 				[0]  number			 no
# user_key 				[1]  string          no
# created_at			[2]  number			 no
# created_str			[3]  date-string     no
# retweet_count			[4]  number          yes
# retweeted				[5]  bool            no
# favorite_count		[6]  number			 yes
# text					[7]  string 		 yes (filter them)
# tweet_id				[8]  number 		 no 
# source				[9]  string 		 no 
# hashtags				[10] string 		 yes (filter them)
# expanded_urls			[11] http-string     yes 
# posted				[12] bool 			 no 
# mentions				[13] string 		 yes (filter them)
# retweeted_status_id	[14] number 		 no 
# in_reply_to_status_id [15] number			 no 

# For testing purposes. The full dataset has 203k tweets 
maxRows = 100000
fractionForLearning = 0.5
numSamples = int(maxRows*fractionForLearning)

russianDataRaw = pd.read_csv("data/Known_Russian_Tweets.csv", 
							 quotechar="\"",
						     header = 0,
						     usecols = (7,),# 4, 6, 10, 11, 13),
						     nrows = maxRows)

# print(russianDataRaw)

# Normal dataset comes with the columns:
# user_id				[0] no 
# tweet_id				[1] no 
# text 					[2] yes 
# date 					[3] no 

normalDataRaw = pd.read_csv("data/Normal_Tweets.txt",
							usecols = (2,),
							sep = "	",
							nrows = maxRows)

normalDataRaw.columns = ['text']

# print(normalDataRaw)

#####################################################
### Building up counts for words in each category ###
#####################################################

NormalDataCounts = dict()
RussianDataCounts = dict()

print("Reading in normal data counts:")
normalErrors = 0
for i in tqdm(range(numSamples)):
	try:
		words = nltk.word_tokenize(normalDataRaw['text'][i])
		for word in nltk.word_tokenize(normalDataRaw['text'][i]):
			if(word.lower() not in StopWords and word.lower() not in StopPunctuation):
				if word.lower() in NormalDataCounts:
					NormalDataCounts[word.lower()] += 1
				else:
					NormalDataCounts[word.lower()] = 1
	except TypeError:
		normalErrors += 1

print("Failed to read " + str(normalErrors) + " lines from normal data")
print("Reading in Russian data counts:")
russianErrors = 0
for i in tqdm(range(numSamples)):
	try:
		for word in nltk.word_tokenize(russianDataRaw['text'][i]):
			if(word.lower() not in StopWords and word.lower() not in StopPunctuation):
				if word.lower() in RussianDataCounts:
					RussianDataCounts[word.lower()] += 1
				else:
					RussianDataCounts[word.lower()] = 1
	except TypeError:
		russianErrors += 1

print("Failed to read " + str(russianErrors) + " lines from normal data")


######################
### Just reporting ###
######################

print("RussianDataCounts: --------------------------------")
for key, value in sorted(RussianDataCounts.items(), key=operator.itemgetter(1)):
	if value > 500:
		print(key + ' ' + str(value))

print("NormalDataCounts: --------------------------------")
for key, value in sorted(NormalDataCounts.items(), key=operator.itemgetter(1)):
	if value > 500:
		print(key + ' ' + str(value))


def LaplaceLikelihood(aWord, aDict, numSamples):
	if (aWord in aDict):
		return (aDict[aWord] + 1.0) / (numSamples + 2.0)
	else:
		return 1.0 / (numSamples + 2.0)

###################
### Predictions ###
###################

numRussSamples = numSamples #int(fractionForLearning*russianDataRaw.shape[0])
numNormSamples = numSamples #int(fractionForLearning*normalDataRaw.shape[0])
totalSamples = numRussSamples + numNormSamples
# Calculate priors up front, they are constant for each prediction:
RussianPrior = numRussSamples / totalSamples
NormalPrior = numNormSamples / totalSamples

russianLabel = 1 
normalLabel = 0

LaplaceFactor = 1

print("Attempting predictions on the remaining normal dataset:")
normalNumCorrect = 0
normalNumIncorrect = 0
normalErrors = 0
for idx in tqdm(range(numNormSamples, normalDataRaw.shape[0])):
	RussianLikelihood = 1
	NormalLikelihood = 1
	try:
		for word in nltk.word_tokenize(normalDataRaw['text'][idx]): #TODO store the filtered data when you gather counts so you don't have to double process things
			lword = word.lower()
			if(lword not in StopWords and lword not in StopPunctuation):
				RussianLikelihood *= LaplaceLikelihood(lword, RussianDataCounts, numRussSamples)
				NormalLikelihood *= LaplaceLikelihood(lword, NormalDataCounts, numNormSamples)

		RussianProb = RussianLikelihood * RussianPrior
		NormalProb = NormalLikelihood * NormalPrior
		if RussianProb > NormalProb:
			normalNumIncorrect += 1 
		else: 
			normalNumCorrect += 1
	except TypeError: 
		normalErrors += 1

accuracyNormal = (normalNumCorrect) / (normalNumCorrect + normalNumIncorrect)*100.0
print ("Accuracy in predicting on the normal dataset: " + str(accuracyNormal))


print("Attempting predictions on the remaining Russian dataset:")
russianNumCorrect = 0
russianNumIncorrect = 0
normalErrors = 0
for idx in tqdm(range(numNormSamples, russianDataRaw.shape[0])):
	RussianLikelihood = 1
	NormalLikelihood = 1
	try:
		for word in nltk.word_tokenize(russianDataRaw['text'][idx]): #TODO store the filtered data when you gather counts so you don't have to double process things
			lword = word.lower()
			if(lword not in StopWords and lword not in StopPunctuation):
				RussianLikelihood *= LaplaceLikelihood(lword, RussianDataCounts, numRussSamples)
				NormalLikelihood *= LaplaceLikelihood(lword, NormalDataCounts, numNormSamples)

		RussianProb = RussianLikelihood * RussianPrior
		NormalProb = NormalLikelihood * NormalPrior
		if RussianProb < NormalProb:
			russianNumIncorrect += 1 
		else: 
			russianNumCorrect += 1
	except TypeError: 
		normalErrors += 1

accuracyRussian = (russianNumCorrect) / (russianNumCorrect + russianNumIncorrect)*100.0
print ("Accuracy in predicting on the russian dataset: " + str(accuracyRussian))