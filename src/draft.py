import pandas as pd
from config import *
from data_util import *
import gensim


# train = pd.read_csv(data_path + 'labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

model = gensim.models.KeyedVectors.load_word2vec_format('D:\PythonProject\model\model_word2vec_google_english\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin', binary= True)
# review, sentiment = load_data_for_text_cnn(data_path + "labeledTrainData.tsv", model, True)
review = load_data_for_text_cnn(data_path + "testData.tsv", model, False)

a = 2