from data_util import *
import gensim
from sklearn.ensemble import RandomForestClassifier
from config import *


train = pd.read_csv(data_path + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] ) )

# 家里
# model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary= False)
# 公司
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary= False)
num_features = 300


# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
# Fit a random forest to the training data, using 100 trees
forest = RandomForestClassifier( n_estimators = 100 )

print("Fitting a random forest to labeled training data...")
forest = forest.fit( trainDataVecs, train["sentiment"] )



# predict
print("Creating average feature vecs for test reviews")
test = pd.read_csv(data_path + "testData.tsv", header=0, delimiter="\t", quoting=3 )
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

# Test & extract results
result = forest.predict( testDataVecs )

# Write the test results
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv(data_path + "Word2Vec_AverageVectors.csv", index=False, quoting=3 )


