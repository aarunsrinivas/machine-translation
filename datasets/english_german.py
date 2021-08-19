import torch
import string
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from util import Vocabulary, SourceTargetPaddingCollate


class EnglishGermanDataset(Dataset):

	def __init__(self, translation_file, freq_threshold=1):
		super(EnglishGermanDataset, self).__init__()
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.df = pd.read_csv(translation_file)
		self.english = self.df['english']
		self.german = self.df['german']
		self.src_vocab = Vocabulary(freq_threshold)
		self.src_vocab.build_vocabulary(self.english.tolist())
		self.tgt_vocab = Vocabulary(freq_threshold)
		self.tgt_vocab.build_vocabulary(self.german.tolist())

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		source = self.english[index]
		target = self.german[index]
		tokenized_source = [self.src_vocab['<sos>']] + self.src_vocab.tokenize(source) + [self.src_vocab['<eos>']]
		tokenized_target = [self.tgt_vocab['<sos>']] + self.tgt_vocab.tokenize(target) + [self.tgt_vocab['<eos>']]
		return torch.tensor(tokenized_source).to(self.device), torch.tensor(tokenized_target).to(self.device)


def load_data(translation_file, freq_threshold=1, batch_size=128, test_size=0.2):
	dataset = EnglishGermanDataset(translation_file, freq_threshold)
	train_size = int((1 - test_size) * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
	train_loader = DataLoader(
		train_dataset,
		batch_size,
		shuffle=True,
		drop_last=True,
		collate_fn=SourceTargetPaddingCollate(dataset.src_vocab, dataset.tgt_vocab)
	)
	test_loader = DataLoader(test_dataset, shuffle=True)
	vocab = {
		'english': dataset.src_vocab,
		'german': dataset.tgt_vocab
	}
	return train_loader, test_loader, vocab
