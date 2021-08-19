import string
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:

	def __init__(self, freq_threshold):
		self.specials = ['<pad>', '<sos>', '<eos>', '<unk>']
		self.freq_threshold = freq_threshold
		self.stoi = {}
		self.itos = []

	def __len__(self):
		return len(self.stoi)

	def __getitem__(self, token):
		return self.stoi[token]

	def build_vocabulary(self, texts):
		texts = map(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)).split(' '), texts)
		vocab = build_vocab_from_iterator(texts, specials=self.specials, min_freq=self.freq_threshold)
		self.itos = vocab.get_itos()
		self.stoi = vocab.get_stoi()

	def tokenize(self, text):
		text = text.lower().translate(str.maketrans('', '', string.punctuation)).split(' ')
		return [self.stoi[token] if token in self.stoi else self.stoi['<unk>'] for token in text]

	def reverse_tokenize(self, sequence):
		sequence = sequence.squeeze(0).detach().numpy()
		bool_array = sequence == self.stoi['<eos>']
		eos_idx = np.where(bool_array)[0][0] if True in bool_array else 1e9
		return ' '.join([self.itos[sequence[i]] for i in range(1, min(eos_idx, len(sequence)))])


class SourceTargetPaddingCollate:

	def __init__(self, src_vocab, tgt_vocab):
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab

	def __call__(self, batch):
		src_batch = [item[0] for item in batch]
		tgt_batch = [item[1] for item in batch]
		src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.src_vocab['<pad>'])
		tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_vocab['<pad>'])
		return src_batch, tgt_batch
