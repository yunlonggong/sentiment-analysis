import pandas as pd
from config import *
from data_util import *
import gensim


# train = pd.read_csv(data_path + 'labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary= True)
review, sentiment = load_data_for_text_cnn(data_path + "labeledTrainData.tsv", model, True)
# review = load_data_for_text_cnn(data_path + "testData.tsv", model, False)

a = 2