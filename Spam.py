import pandas as pd
import string
import pprint
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


with open('SMSSpamCollection.csv') as f:
    df = pd.read_table(f, sep='\t', header=None, names=["label",'sms_message'],
                          lineterminator='\n')


df["label"] = df.label.map({"ham": 0, "spam": 1})

# df['label'] = map(df,di)

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']


#############count vector from scratch##########
lower_case_documents = []
sans_punctuation_documents = []
frequency_list = []
frequency_list = Counter()
for element in documents:
    	lower_case_documents.append(element.lower())

for i in lower_case_documents:
	sans_punctuation_documents.append(i.translate(str.maketrans('','',string.punctuation)))
for i in lower_case_documents:
	frequency_list.update(Counter(i.split()))  ##not completely correct needs to append all results
#pprint.pprint(frequency_list) 
#print(sans_punctuation_documents)

# print(df.head(5))
# print(table[1:6,])    

############## Done scratching back to real world##################


count_vector= CountVectorizer(input=documents, encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.int64)
count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray() #matrix creation then change to array

print(doc_array)

