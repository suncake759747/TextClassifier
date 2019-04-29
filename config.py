batch_size=64 # batch size for training
dropout_rate=0.2 # dropout rate (for all dropouts in training)
embedding_size=100 # dimensionality of embeddings
max_len=400 # Max document length (shorter docs will be padded, longer ones truncated)
min_word_count_freq = 4 # Min frequency of word in corpus to be used
num_epochs=2 # number of training epochs