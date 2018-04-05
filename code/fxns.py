import numpy as np
import pandas as pd
import collections
import os.path, pdb
import tensorflow as tf

def create_shards(path, train_out_path, valid_out_path, test_out_path,
        test_chrs, valid_chrs,
        blocksize=1000, text=True, binary=True, num_blocks=None):
    # output 2 folders, with samples from 

    from os import listdir
    from os.path import isfile, join
    files = [join(path,f) for f in listdir(path) if isfile(join(path, f))]

    # figure out one hot encoding mapping
    import string
    tf_d = {}
    train_chr_to_files = collections.defaultdict(list)
    valid_chr_to_files = collections.defaultdict(list)
    test_chr_to_files = collections.defaultdict(list)
    for f in files:
        _, name = os.path.split(f)
        tft, chr, _ = string.split(name, sep='.')
        chr = int(chr)
        tf_d[tft] = 1
        if chr in test_chrs:
            test_chr_to_files[chr].append(f)
        elif chr in valid_chrs:
            valid_chr_to_files[chr].append(f)
        else:
            train_chr_to_files[chr].append(f)

    tf_to_pos = dict(map(reversed,enumerate(tf_d.keys())))

    # create iterator over lines

    def file_input(files):
        for f in files:
            with open(f,'r') as fin:
                for line in fin:
                    yield line

    train_readers = {chr:file_input(train_chr_to_files[chr]) for chr in train_chr_to_files.keys()}
    valid_readers = {chr:file_input(valid_chr_to_files[chr]) for chr in valid_chr_to_files.keys()}
    test_readers = {chr:file_input(test_chr_to_files[chr]) for chr in test_chr_to_files.keys()}

    seq_len = None
    label_len = None

    # read lines from iterators and write, removing readers if they are empty
    def write(readers, blocksize, path):

        nuc_to_onehot_d = {
            'A':[1,0,0,0],
            'G':[0,1,0,0],
            'T':[0,0,1,0],
            'C':[0,0,0,1],
            }

        def nuc_to_onehot(c):
            try:
                return nuc_to_onehot_d[c]
            except KeyError:
                return [0,0,0,0]

        if not os.path.exists(path):
            os.makedirs(path)
        idx = 0
        this_idx = 0
        text_writer = None
        binary_writer = None
        while True:
            if not (num_blocks is None) and idx > num_blocks:
                break
            if len(readers) == 0:
                break
            if text_writer is None:
                if text:
                    text_writer = open('%s/shard_%d.txt' % (path, idx), 'w')
            if binary_writer is None:
                if binary:
                    binary_writer = tf.python_io.TFRecordWriter('%s/shard_%d.tfrecords' % (path, idx))
            key = np.random.choice(readers.keys())
            try:
                line = readers[key].next()
            except StopIteration:
                del readers[key]
            else:
                tft, seq = string.split(line.strip(), sep='\t')
                seq_onehot = np.array([nuc_to_onehot(c) for c in seq]).reshape(-1)
                seq_len = len(seq_onehot)
                tf_onehot = np.zeros(len(tf_to_pos), dtype=int)
                tf_onehot[tf_to_pos[tft]] = 1
                label_len = len(tf_onehot)
                if text:
                    text_writer.write(string.join(map(str, seq_onehot) + map(str, tf_onehot), sep=' ') + '\n')
                if binary:
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=tf_onehot)),
                                'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=seq_onehot)),
                                }
                            )
                        )
                    binary_writer.write(example.SerializeToString())
                this_idx += 1
            if this_idx % blocksize == 0:

                this_idx = 0
                idx += 1
                print idx

                if text:
                    text_writer.close()
                    text_writer = None
                if binary:
                    binary_writer.close()
                    binary_writer = None

        pd.Series({'tf_to_pos':tf_to_pos, 'seq_len':seq_len, 'label_len':label_len}).to_csv('%s/info' % path)

    write(train_readers, blocksize, train_out_path)
    write(valid_readers, blocksize, valid_out_path)
    write(test_readers, blocksize, test_out_path)

def get_seq_and_label(out_path):
    from os import listdir
    from os.path import isfile, join
    files = [join(out_path,f) for f in listdir(out_path) if isfile(join(out_path, f)) and f[-9:] == 'tfrecords']
    filename_queue = tf.train.string_input_producer(files, num_epochs=None)
    reader = tf.TFRecordReader()
    info = pd.Series.from_csv('%s/info' % out_path)
    info['seq_len'] = int(info['seq_len'])
    info['label_len'] = int(info['label_len'])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([info['label_len']], tf.int64),
            'seq': tf.FixedLenFeature([info['seq_len']], tf.int64)
            })
    label = tf.cast(features['label'], tf.float32)
    seq = tf.cast(features['seq'], tf.float32)

    return (seq, label), info

def get_logreg_model(seq_batch, label_batch, seq_len, label_len):
    B = tf.get_variable("B", [seq_len, label_len], dtype=tf.float32)
    logits = tf.matmul(seq_batch, B)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=logits), axis=[0,1])
    return loss
