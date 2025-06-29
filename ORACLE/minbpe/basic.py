from tqdm import tqdm
from .base import Tokeniser, get_stats, merge


class BasicTokeniser(Tokeniser):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for i in tqdm(range(num_merges), total = num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key = stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors = "replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids