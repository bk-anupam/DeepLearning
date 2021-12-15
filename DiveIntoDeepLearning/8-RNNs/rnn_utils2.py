import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import re
from collections import Counter

# Read the data ("The time machine")
def read_text_file(file_path):
    lines = []
    with open(file_path, mode='r') as f:
        lines = f.readlines()
        
    return [re.sub("[^A-Za-z|\s]", "", line).strip().lower() for line in lines]

# The following tokenize function takes a list (lines) as the input, where each element is a text sequence 
# (e.g., a text line). Each text sequence is split into a list of tokens. A token is the basic unit in
#  text. In the end, a list of token lists are returned, where each token is a string.
def tokenize(lines, token="word"):    
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]

def count_token_freq(tokens):
    # flatten the list of lists
    tokens = [token for line in tokens for token in line]     
    return Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, reserved_tokens=None, min_freq=0, ):
        if tokens == None:
            tokens = []
        if reserved_tokens == None:
            reserved_tokens = []
        self._token_freq = {}
        if len(tokens) > 0:
            self._token_freq = count_token_freq(tokens)           
        # Sort the tokens by their frequencies in descending order            
        self._sorted_token_freq = [(value, key) for key, value in self._token_freq.items()]
        self._sorted_token_freq.sort(reverse=True, key=lambda k: k[0])
        # The unknown token followed by reserved tokens are at the begining
        self.idx_to_token = ['unk'] + reserved_tokens
        self.token_to_idx = {token: index for index, token in enumerate(self.idx_to_token)}
        # iterate thru the sorted tokens list, if token not present in idx_to_token append at the 
        # end of the list. If token freq is less than min_freq, skip the token
        for freq, token in self._sorted_token_freq:
            if freq < min_freq:
                continue
            if token not in self.token_to_idx.keys():
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    # the index of <unk> token
    @property
    def unk(self):
        return 0

    # tokens sorted by their frequencies
    @property
    def token_freq(self):
        return self._sorted_token_freq

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # if tokens is not an instance of list of tuple i.e. we want index of a single token
        if not isinstance(tokens, (list, tuple)):
            # if tokens is not present in token_to_idx dict, return the index of unk token
            return self.token_to_idx.get(tokens, self.unk)
        # return index of multiple tokens
        return [self.token_to_idx[token] for token in tokens]

    def to_tokens(self, indices):
        # if indices is not an instance of list of tuple i.e. we want token of a single index
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # return tokens for multiple indices            
        return [self.idx_to_token[index] for index in indices]         

def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_text_file("./timemachine.txt")
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a sequence
    corpus = corpus[np.random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    np.random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for subsequences
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = np.random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y        

class SeqDataLoader:  
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)        

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab        