# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import tensorflow as tf
from text_cnn_model import TextCNN
import os
from data_util import *
import gensim
from config import *

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("traning_data_path","../data/sample_multiple_label.txt","path of traning data.") #sample_multiple_label.txt-->train_label_single100_merge
tf.app.flags.DEFINE_integer("vocab_size",400000,"maximum vocab size.")

tf.app.flags.DEFINE_float("learning_rate",0.0003,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir",data_path + "text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",MAX_LENGTH,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",50,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512
tf.app.flags.DEFINE_string("word2vec_model_path", word2vec_path, "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_integer("num_classes",2,"num_classes of output.")
tf.app.flags.DEFINE_float("rate_data_train", 0.9, "rate of data to train, for example, 0.9 of the train data to train, and 0.1 of the train data to test when training.")
filter_sizes=[6,7,8]

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.word2vec_model_path, binary=False)

    num_classes = FLAGS.num_classes
    testX = load_data_for_text_cnn(data_path + "testData.tsv", word2vec_model, is_training=False)
    #print some message for debug purpose
    print("length of test data:",len(testX))
    print("testX[0]:", testX[0])

    #2.create session.
    # config=tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        #Instantiate Model
        textCNN = TextCNN(filter_sizes,FLAGS.num_filters,num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,FLAGS.vocab_size,FLAGS.embed_size,FLAGS.is_training)
        #Initialize Save
        saver=tf.train.Saver()
        if not os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("checkpoint file not exists " + FLAGS.ckpt_dir)

        print("Restoring Variables from Checkpoint.")
        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))

        number_of_test_data = len(testX)
        for start, end in zip(range(0, number_of_test_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_test_data+1, FLAGS.batch_size)):
            logits=sess.run(textCNN.logits,feed_dict={textCNN.input_x:testX[start:end],textCNN.dropout_keep_prob:1})
            a= 1

    pass


if __name__ == "__main__":
    tf.app.run()