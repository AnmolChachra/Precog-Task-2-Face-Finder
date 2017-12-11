import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit
import time
import datetime
import MyCNN

ARVIND_K_PICS = r'ImageTask/Arvind_Kejriwal_Pics/OnlyFaceAligned/'
NARENDRA_M_PICS = r'ImageTask/Narendra_Modi Pics/OnlyFaceAligned/'
RANDOM_PICS = r'ImageTask/Random Face Pics/OnlyFaceAligned/'

"""
This Function will create Training and Testing Data

"""
def create_train_test_data():
    X_Data = []
    Y_Data = [] #{1:'Arvind Kejriwal',2:'Narendra Modi',3:'Other'}
    for img in os.listdir(r'ImageTask/Arvind_Kejriwal_Pics/OnlyFaceAligned/'):
        path = os.path.join(ARVIND_K_PICS,img)
        image = cv2.imread(path)
        image = cv2.resize(image,(100,100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_Data.append(image.astype(np.float32))
        Y_Data.append(np.array([1,0,0],dtype=np.int32))
      
    
    for img in os.listdir(NARENDRA_M_PICS):
        path = os.path.join(NARENDRA_M_PICS,img)
        image = cv2.imread(path)
        image = cv2.resize(image,(100,100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_Data.append(image.astype(np.float32))
        Y_Data.append(np.array([0,1,0],dtype=np.int32))
        
    for img in os.listdir(RANDOM_PICS):
        path = os.path.join(RANDOM_PICS,img)
        image = cv2.imread(path)
        image = cv2.resize(image,(100,100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_Data.append(image.astype(np.float32))
        Y_Data.append(np.array([0,0,1],dtype=np.int32))

    return np.array(X_Data,dtype=np.float32), np.array(Y_Data,dtype=np.int32)

X,Y = create_train_test_data()

X = X.reshape((-1,100,100,1))

def shuffle_and_split(X_data, Y_data, random_seed):
    rs = ShuffleSplit(n_splits=1,train_size=0.9,test_size=0.1,random_state=random_seed)
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    for trainX,testX in rs.split(X_data):
        for i in trainX:
            Xtrain.append(X[i])
        for i in testX:
            Xtest.append(X[i])
    for trainY,testY in rs.split(Y_data):
        for i in trainY:
            Ytrain.append(Y[i])
        for i in testX:
            Ytest.append(Y[i])
    return np.array(Xtrain,dtype=np.float32),np.array(Ytrain,dtype=np.int32),np.array(Xtest, dtype=np.float32),np.array(Ytest, dtype=np.int32)

Xtrain, Ytrain, Xtest, Ytest = shuffle_and_split(X,Y,1)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

#Hyperparameters
tf.flags.DEFINE_integer("batch_size", 33, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 1000)")
tf.flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on test set after this many steps (default: 1000)")
tf.flags.DEFINE_integer("checkpoint_every", 20, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("Parameters:\n")

for attr, value in sorted(FLAGS.__flags.items()):
    print("{0} = {1}".format(attr.upper(), value))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = MyCNN.CNN(image_dimensions=(Xtrain.shape[1],Xtrain.shape[2],Xtrain.shape[3]),num_classes=Ytrain.shape[1],filter_size=4,num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        lr = tf.train.exponential_decay(0.005,
                                  global_step,
                                  50,
                                  0.5,
                                  staircase=True)     
        optimizer = tf.train.AdamOptimizer(lr)
        
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            print("here")
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = batch_iter(list(zip(Xtrain, Ytrain)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(Xtest, Ytest, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
