from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from data_util import *
import pandas as pd
from config import *


train = pd.read_csv(data_path + 'labeledTrainData.tsv', header=0,
                delimiter="\t", quoting=3)
test = pd.read_csv(data_path + 'testData.tsv', header=0, delimiter="\t",
                quoting=3 )

traindata = []
for i in range(0,len(train['review'])):
    traindata.append(" ".join(review_to_wordlist(train['review'][i])))

testdata = []
for i in range(0,len(test['review'])):
    testdata.append(" ".join(review_to_wordlist(test['review'][i])))

tfv = TFIV(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

X_all = traindata + testdata # Combine both to fit the TFIDF vectorization.
lentrain = len(traindata)

tfv.fit(X_all) # This is the slow part!
X_all = tfv.transform(X_all)

X = X_all[:lentrain] # Separate back into training and test sets.
X_test = X_all[lentrain:]

words = tfv.get_feature_names()

# 参考如何打印
i = 0
for j in range(len(words)):
    if X[i,j] > 1e-5:
          print( words[j], X[i,j])

print(words[304735])
print(X[0,304735])

print("finished.")