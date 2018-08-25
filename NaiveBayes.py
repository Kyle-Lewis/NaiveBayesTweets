#NaiveBayes.py
from h2o.estimators import H2ONaiveBayesEstimator
import nltk 

#from BayesData import BayesData 
#from Tokenizer import Tokenizer 

import numpy as np 
import pandas as pd
from collections import namedtuple 

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
maxRows = 5000
russianDataRaw = pd.read_csv("data/Known_Russian_Tweets.csv", 
							 quotechar="\"",
						     header = 0,
						     usecols = (7,),# 4, 6, 10, 11, 13),
						     nrows = maxRows)

print(russianDataRaw)

# Normal dataset comes with the columns:
# user_id				[0] no 
# tweet_id				[1] no 
# text 					[2] yes 
# date 					[3] no 

normalDataRaw = pd.read_csv("data/Normal_Tweets.txt",
							usecols = (2,),
							sep = "	",
							nrows = maxRows)

print(normalDataRaw)
# print(russianDataRaw.shape[0])
print(str(russianDataRaw["text"][0]))
for i in range(russianDataRaw.shape[0]-1):
	print(nltk.word_tokenize(russianDataRaw["text"][i]))


