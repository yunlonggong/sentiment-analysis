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

# tf.app.flags.DEFINE_string("traning_data_path","../data/sample_multiple_label.txt","path of traning data.") #sample_multiple_label.txt-->train_label_single100_merge
tf.app.flags.DEFINE_integer("vocab_size",400000,"maximum vocab size.")

tf.app.flags.DEFINE_float("learning_rate",0.0003,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir",data_path + "text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",MAX_LENGTH,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",50,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
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
    trainX, trainY, testX, testY = None, None, None, None
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.word2vec_model_path, binary=False)
    # word2vec_index2word = word2vec_model.index2word
    # vocab_size = len(word2vec_index2word)


    num_classes = FLAGS.num_classes
    review, sentiment = load_data_for_text_cnn(data_path + "labeledTrainData.tsv", word2vec_model, is_training=True)
    trainX, trainY = review[0 : int(len(review) * FLAGS.rate_data_train)], sentiment[0 : int(len(review) * FLAGS.rate_data_train)]
    testX, testY = review[int(len(review) * FLAGS.rate_data_train) : len(review)], sentiment[int(len(review) * FLAGS.rate_data_train) : len(review)]
    #print some message for debug purpose
    print("length of training data:",len(trainX),";length of validation data:",len(testX))
    print("trainX[0]:", trainX[0])
    print("trainY[0]:", trainY[0])
    train_y_short = get_target_label_short(trainY[0])
    print("train_y_short:", train_y_short)

    #2.create session.
    # config=tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        #Instantiate Model
        textCNN = TextCNN(filter_sizes,FLAGS.num_filters,num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,FLAGS.vocab_size,FLAGS.embed_size, is_training=True)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            #for i in range(3): #decay learning rate if necessary.
            #    print(i,"Going to decay learning rate by half.")
            #    sess.run(textCNN.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess,textCNN,word2vec_model)
        curr_epoch=sess.run(textCNN.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        iteration=0
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, counter =  0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration=iteration+1
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
                feed_dict = {textCNN.input_x: trainX[start:end],textCNN.dropout_keep_prob: 0.5,textCNN.iter: iteration,textCNN.tst: not FLAGS.is_training}
                feed_dict[textCNN.input_y] = trainY[start:end]
                curr_loss,lr,_,_=sess.run([textCNN.loss_val,textCNN.learning_rate,textCNN.update_ema,textCNN.train_op],feed_dict)
                loss,counter=loss+curr_loss,counter+1
                if counter %50==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),lr))

                ########################################################################################################
                if start%(2000*FLAGS.batch_size)==0: # eval every 3000 steps.
                    eval_loss, f1_score, precision, recall = do_eval(sess, textCNN, testX, testY,iteration)
                    print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f" % (epoch, eval_loss, f1_score, precision, recall))
                    # save model to checkpoint
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    saver.save(sess, save_path, global_step=epoch)
                ########################################################################################################
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss,f1_score,precision,recall=do_eval(sess,textCNN,testX,testY,iteration)
                print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f" % (epoch,eval_loss,f1_score,precision,recall))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss,_,_,_ = do_eval(sess, textCNN, testX, testY,iteration)
        print("Test Loss:%.3f" % ( test_loss))
    pass


# 在验证集上做验证，报告损失、精确度
def do_eval(sess,textCNN,evalX,evalY,iteration):
    number_examples=len(evalX)
    eval_loss,eval_counter,eval_f1_score,eval_p,eval_r=0.0,0,0.0,0.0,0.0
    batch_size=1
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y:evalY[start:end],textCNN.dropout_keep_prob: 1.0,textCNN.iter: iteration,textCNN.tst: True}
        curr_eval_loss, logits= sess.run([textCNN.loss_val,textCNN.logits],feed_dict)#curr_eval_acc--->textCNN.accuracy
        label_list_top5 = get_label_using_logits(logits[0])
        f1_score,p,r=compute_f1_score(list(label_list_top5), evalY[start:end][0])
        eval_loss,eval_counter,eval_f1_score,eval_p,eval_r=eval_loss+curr_eval_loss,eval_counter+1,eval_f1_score+f1_score,eval_p+p,eval_r+r
    return eval_loss/float(eval_counter),eval_f1_score/float(eval_counter),eval_p/float(eval_counter),eval_r/float(eval_counter)

def compute_f1_score(label_list_top5,eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    num_correct_label=0
    eval_y_short=get_target_label_short(eval_y)
    for label_predict in label_list_top5:
        if label_predict in eval_y_short:
            num_correct_label=num_correct_label+1
    #P@5=Precision@5
    num_labels_predicted=len(label_list_top5)
    all_real_labels=len(eval_y_short)
    p_5=num_correct_label/num_labels_predicted
    #R@5=Recall@5
    r_5=num_correct_label/all_real_labels
    f1_score=2.0*p_5*r_5/(p_5+r_5+0.000001)
    return f1_score,p_5,r_5

def get_target_label_short(eval_y):
    eval_y_short=[] #will be like:[22,642,1391]
    for index,label in enumerate(eval_y):
        if label>0:
            eval_y_short.append(index)
    return eval_y_short

#get top5 predicted labels
def get_label_using_logits(logits,top_number=5):
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1
    return count / len(labels)

def assign_pretrained_word_embedding(sess,textCNN,word2vec_model):
    word2vec_index2word = word2vec_model.index2word
    vocab_size = len(word2vec_index2word)
    word2vec_dict = {}
    for word in word2vec_model.vocab:
        word2vec_dict[word] = word2vec_model[word]
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = word2vec_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    tf.app.run()
