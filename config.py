batch_size=64 # batch size for training
dropout_rate=0.1 # dropout rate (for all dropouts in training)
embedding_size=50 # dimensionality of embeddings
max_len=400 # Max document length (shorter docs will be padded, longer ones truncated)
min_word_count_freq = 4 # Min frequency of word in corpus to be used
num_epochs=2 # number of training epochs