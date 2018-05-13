from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np  # Make sure that numpy is imported
import pandas as pd


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


# todo, 返回值的类型需要修改成二维的形式
def load_data_for_text_cnn(data_path_name, model, is_training):
    '''
    :param data_path_name: can be train data or test data, path and name
    :param model: word2vec_model
    :param is_training: True for train, return review and sentiment; False for test, return review;
    :return: dataX for nn input and dataY for nn output
    '''
    data = pd.read_csv(data_path_name, header=0, delimiter="\t", quoting=3)
    # Get the number of reviews based on the dataframe column size
    num_reviews = data["review"].size

    # Initialize an empty list to hold the clean reviews
    data_review = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    for i in range(0, num_reviews):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        clean_review = review_to_words(data["review"][i]).split()
        index_of_words = []
        for j in range(0, len(clean_review)):
            if clean_review[j] in model:
                index_of_words.append(model.vocab[clean_review[j]].index)
        data_review.append(index_of_words)

    # for test data, has no sentiment
    if is_training == False:
        return data_review

    # for train data, has sentiment
    data_sentiment = []
    for i in range(0, num_reviews):
        data_sentiment.append(data["sentiment"][i])

    # if return data is wrong, throw error(exception)
    if len(data_review) != len(data_sentiment):
        raise RuntimeError("wrong when load train data, review size does not equal to sentiment size")

    return data_review, data_sentiment