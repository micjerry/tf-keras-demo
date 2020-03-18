import os
import numpy as np
import csv
import string
import requests
import io
from zipfile import ZipFile
import itertools
from collections import Counter


def load_data_and_labels():
    save_file_name = os.path.join('temp','temp_spam_data.csv')

    if os.path.isfile(save_file_name):
        text_data = []
        with open(save_file_name, 'r') as temp_output_file:
            reader = csv.reader(temp_output_file)
            for row in reader:
                text_data.append(row)
    else:
        zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        r = requests.get(zip_url)
        z = ZipFile(io.BytesIO(r.content))
        file = z.read('SMSSpamCollection')
        # Format Data
        text_data = file.decode()
        text_data = text_data.encode('ascii',errors='ignore')
        text_data = text_data.decode().split('\n')
        text_data = [x.split('\t') for x in text_data if len(x)>=1]
        # And write to csv
        with open(save_file_name, 'w') as temp_output_file:
            writer = csv.writer(temp_output_file)
            writer.writerows(text_data)
        
    texts = [x[1] for x in text_data]
    target = [x[0] for x in text_data]
    # Relabel 'spam' as 1, 'ham' as 0
    target = [[0,1] if x=='spam' else [1,0] for x in target]

    # Convert to lower case
    texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    
    texts = [s.split(" ") for s in texts]

    return [texts, target]

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def load_data():
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]