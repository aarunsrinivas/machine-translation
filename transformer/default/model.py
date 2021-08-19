import torch
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):

	def __init__(self, d_model, max_seq_len=100):
		super(PositionalEncoding, self).__init__()
		self.d_model = d_model
		self.pe = torch.zeros(max_seq_len, d_model, requires_grad=False)
		for pos in range(max_seq_len):
			for i in range(0, d_model, 2):
				self.pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
				self.pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
		self.pe = self.pe.unsqueeze(0)
		self.register_buffer('pe', self.pe)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x):
		x = x * np.sqrt(self.d_model)
		seq_len = x.shape[1]
		x = x + self.pe[:, :seq_len].to(self.device)
		return x


class Transformer(nn.Module):

	def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, num_layers, p):
		super(Transformer, self).__init__()
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.src_embedding = nn.Embedding(len(src_vocab), d_model)
		self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model)
		self.positional_encoding = PositionalEncoding(d_model)
		self.transformer = nn.Transformer(d_model, num_heads, num_layers, num_layers, 4, batch_first=True)
		self.dropout = nn.Dropout(p)
		self.linear = nn.Linear(d_model, len(tgt_vocab))
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, src, tgt):
		src_embedding = self.dropout(self.positional_encoding(self.src_embedding(src)))
		tgt_embedding = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))
		src_key_padding_mask = src == self.src_vocab['<pad>']
		tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)
		transformer_out = self.transformer(src_embedding, tgt_embedding, tgt_mask=tgt_mask,
		                                   src_key_padding_mask=src_key_padding_mask)
		out = self.linear(transformer_out)
		return out

	def translate_sequence(self, src_sequence, tgt_max_len=100):
		tgt_sequence = torch.zeros(1, tgt_max_len, dtype=torch.int32).to(self.device)
		tgt_sequence[0][0] = self.tgt_vocab['<sos>']
		for i in range(1, tgt_max_len):
			outputs = torch.argmax(self.forward(src_sequence, tgt_sequence), dim=-1)[0]
			prediction = outputs[i - 1].item()
			if prediction == self.tgt_vocab['<eos>']:
				break
			tgt_sequence[0][i] = prediction
		return self.tgt_vocab.reverse_tokenize(tgt_sequence)

