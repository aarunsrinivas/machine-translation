import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):

	def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, p):
		super(Encoder, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, batch_first=True)
		self.dropout = nn.Dropout(p)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x):
		embedding = self.dropout(self.embedding(x))
		outputs, (hidden, cell) = self.lstm(embedding)
		return hidden, cell


class Decoder(nn.Module):

	def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers, p):
		super(Decoder, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, batch_first=True)
		self.linear = nn.Linear(hidden_size, output_size)
		self.dropout = nn.Dropout(p)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x, hidden, cell):
		embedding = self.dropout(self.embedding(x))
		outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
		predictions = self.linear(outputs)
		return predictions, hidden, cell


class Seq2Seq(nn.Module):

	def __init__(self, src_vocab, tgt_vocab, embedding_size, hidden_size, num_layers, p):
		super(Seq2Seq, self).__init__()
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.encoder = Encoder(len(src_vocab), embedding_size, hidden_size, num_layers, p)
		self.decoder = Decoder(len(tgt_vocab), embedding_size, hidden_size, len(tgt_vocab), num_layers, p)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, src, tgt, teacher_force=0.5):
		batch_size = tgt.shape[0]
		tgt_length = tgt.shape[1]
		outputs = torch.zeros(batch_size, tgt_length, len(self.tgt_vocab)).to(self.device)
		hidden, cell = self.encoder(src)
		x = tgt[:, 0].unsqueeze(1)
		for i in range(1, tgt_length):
			output, hidden, cell = self.decoder(x, hidden, cell)
			outputs[:, i, :] = output.squeeze(1)
			prediction = output.argmax(-1)
			x = tgt[:, i].unsqueeze(1) if np.random.random() < teacher_force else prediction
		return outputs

	def translate_sequence(self, src_sequence, tgt_max_len=100):
		outputs = self.forward(src_sequence,
		                       torch.zeros(1, tgt_max_len, dtype=torch.int32).to(self.device), teacher_force=0.0)
		tgt_sequence = torch.argmax(outputs, dim=-1)[0][1:]
		return self.tgt_vocab.reverse_tokenize(tgt_sequence)
