import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import random


path = os.path.join(os.path.sep, 'Users', 'egilbertson', 'Box', 'UCSF','Winter2020','Algorithms','Final_Project_Skeleton','data')


def to_binary(seqs):
    """

    Input: a list of sequences (base pair letter strings)
    Output: a list of binarized sequences

    takes in list of sequences, returns list of sequences in binary

    binarize using 6 positions, 1 for each base pair and an additional 2 to encode purine vs pyrimadine for additional
    information on transitions/transversions to be included in the encoding


    """
    bin_dict = dict()
    bin_dict["A"] = "100010"
    bin_dict["C"] = "010001"
    bin_dict["G"] = "001010"
    bin_dict["T"] = "000101"
    bin_seqs = []
    for seq in seqs:
        bin_seq = ""
        for char in seq:
            bin_seq += bin_dict[char]
        bin_seqs.append(bin_seq)

    return bin_seqs


def string_to_array(binary_strings):
    """
    Input: list of binary sequences
    Output: numpy array with same information, each column is a single bit of the binarized sequence
    """
    rows = len(binary_strings)
    cols = len(binary_strings[0])

    seq_array = np.zeros((rows, cols), dtype=int)

    for i, seq in enumerate(binary_strings):
        for j, char in enumerate(seq):
            seq_array[i, j] = int(char)

    return seq_array


def read_training_data(file_path, filename, class_size):
    """
    Input: file path string, filename string and size of class
    Output: array of sequnces and array of classes
    """
    sequences = []
    classes  = []

    with open(os.path.join(file_path, filename)) as f:
        for line in f:
            l = line.strip()
            classes.append(l[:class_size])
            sequences.append(l[class_size:])

    return (string_to_array(sequences), string_to_array(classes))



def reverse_complement(seqs):
    '''
    Input: list of sequences
    Output: set of reverse complements of the input sequences
    '''
    reverse_set = set()
    for item in seqs:
        tmp = Seq(item)
        reverse_set.add(tmp.reverse_complement)
    return reverse_set


def master_training_set(pos, neg):
    """
    Input: positive and negatice sequences in binary form
    Output: set of positive and negative sequences to be used in training the neural net
    """
    np.random.seed(100)  # set random seed so results can be reproduced
    master = set()

    for posseq in pos:
        tmp = '01' + posseq # add '01' at beginning to identify positive sequences
        master.add(tmp)

    number_of_neg_examples = len(pos) * 100 # take 100x more negative samples than positive

    random.shuffle(neg) # randomly shuffle the negative set so choice can be a one liner

    for negseq in neg[0:number_of_neg_examples]: # choose first n negative samples after reshuffling
        tmp = '10' + negseq # '10' to ID negative sequences
        master.add(tmp)

    return master

def write_output(pos, neg, master):
    """
    writes output files
    training_pos = all positive data
    training_neg = all negative data
    training = master training set to be used
    """

    with open(os.path.join(file_path, "training_pos.txt"), 'w+') as outset:
        # write the sets
        outset.write('\n'.join(x for x in pos))

    with open(os.path.join(file_path, "training_neg.txt"), 'w+') as outset:
        # write the sets
        outset.write('\n'.join(x for x in neg))

    with open(os.path.join(file_path, "training.txt"), 'w+') as outset:
        outset.write('\n'.join(x for x in master))


def read_positives():
    '''
    Input: nothing
    Output: list of sequences in the positive seqs file
    '''
    fn = 'rap1-lieb-positives.txt'
    file = open(os.path.join(path, fn))
    seqs = []
    for line in file:
        seqs.append(line.strip())

    return seqs

def read_negatives(pos_seqs):
    '''
    Input: list of positive sequences
    Output: list of true negative sequences

    ** have to check for positives within the negative sequences
    '''
    pos_set = set(pos_seqs) # cast as set in order to prevent duplicates
    p_reverse = reverse_complement(pos_set) # get reverse complements
    all_pos = pos_set.union(p_reverse) # all positives include normal and rev_comp

    fn = 'yeast-upstream-1k-negative.fa'
    neg_file = SeqIO.parse(open(os.path.join(path, fn)), 'fasta')
    n_seqs = []
    for seq in neg_file:
        n_seqs.append(seq.seq)

    real_negs = set()
    for s in n_seqs: # check that negative seq is not replicated in positive set, don't include it if it is
        for i in range(len(s) - 16):
            split_s = s[i:i + 17]
            if split_s not in all_pos:
                real_negs.add(str(split_s))


    return list(real_negs)
