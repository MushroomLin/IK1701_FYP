#! /usr/bin/env python
# Modified from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/eval.py

import tensorflow as tf
import numpy as np
import os
import data_function as df
from tensorflow.contrib import learn
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import csv
from data_function import tfidf_score

def test_file_loader(file_path):
    with open(file_path,'r') as f:
        sentences=[]
        num=0
        for line in f:
            sentences.append(line)
            num+=1
        f.close()
    return num, sentences


# Parameters
# ==================================================


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1500736434/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

test_file='./test/doc/1'
dir_path='./test/doc'
file_num=1
total_num, x_raw = test_file_loader(test_file)
print(total_num)
y_test = np.zeros(total_num)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = df.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

tfidf_df, max_score = tfidf_score(file_num=file_num,dir_path=dir_path,file_path=test_file)
print("max score",max_score)
tfidf_df.sort_values(by='score',ascending=False).to_csv('./tfidf_score.csv')

count=0
if y_test is not None:
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    print(predictions_human_readable.shape)
    for i in predictions_human_readable:
        if float(i[1])==1:
            tfidf_df.loc[count,'score']=tfidf_df.loc[count,'score']/max_score+1
        else:
            tfidf_df.loc[count, 'score'] = tfidf_df.loc[count, 'score'] / max_score
        count+=1

tfidf_df=tfidf_df.sort_values(by='score',ascending=False)
tfidf_df.to_csv('./adjusted_score.csv')
# Save the evaluation to a csv
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)