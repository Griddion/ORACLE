from minbpe import BasicTokeniser

with open("corpus data/training_corpus.txt", "r", encoding = "utf-8") as f:
    text_sequence = f.read()

length = len(text_sequence)
print(f"corpus length: {length}")


tokeniser = BasicTokeniser()
tokeniser.train(text_sequence, vocab_size = 1024)

enc = tokeniser.encode("bonjour, çava?")
print(enc)

dec = tokeniser.decode([98, 111, 110, 106, 111, 117, 114, 44, 32, 195, 167, 97, 118, 97, 63])
print(dec)

max_vocab_id = list(tokeniser.vocab.keys())[-1]
tokeniser.special_tokens = {
    "<|startoftext|>": max_vocab_id + 1,
    "<|separator|>": max_vocab_id + 2,
    "<|endoftext|>": max_vocab_id + 3,
    "<|unk|>": max_vocab_id + 4,
    "<|padding|>": max_vocab_id + 5,
}

length = len(tokeniser.encode(text_sequence))
print(length)

tokeniser.save(file_prefix="output/tokeniser/my_tokeniser")


