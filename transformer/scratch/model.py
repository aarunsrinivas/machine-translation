import torch
from torch import nn
import torch.nn.functional as F
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
		self.register_buffer('positional_encoding', self.pe)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x):
		x = x * np.sqrt(self.d_model)
		seq_len = x.shape[1]
		x = x + self.pe[:, :seq_len].to(self.device)
		return x


class MultiHeadAttention(nn.Module):

	def __init__(self, d_model, num_heads, p=0.1):
		super(MultiHeadAttention, self).__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.query_size = d_model // num_heads
		self.q_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model)
		self.v_linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(p)
		self.out = nn.Linear(d_model, d_model)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, q, k, v, mask=None):
		batch_size = q.shape[0]
		q = self.q_linear(q).reshape(batch_size, -1, self.num_heads, self.query_size).transpose(1, 2)
		k = self.k_linear(k).reshape(batch_size, -1, self.num_heads, self.query_size).transpose(1, 2)
		v = self.v_linear(v).reshape(batch_size, -1, self.num_heads, self.query_size).transpose(1, 2)
		scores = self.attention(q, k, v, mask=mask)
		concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
		output = self.out(concat)
		return output

	def attention(self, q, k, v, mask=None):
		k_t = k.transpose(2, 3)
		scores = torch.matmul(q, k_t) / np.sqrt(self.d_model)
		if mask is not None:
			mask = mask.unsqueeze(1)
			scores = scores.masked_fill(mask == 0.0, -1e9)
			scores = F.softmax(scores, dim=-1)
		scores = self.dropout(scores)
		output = torch.matmul(scores, v)
		return output


class FeedForwardBlock(nn.Module):

	def __init__(self, d_model, forward_expansion=2048, p=0.1):
		super(FeedForwardBlock, self).__init__()
		self.block = nn.Sequential(
			nn.Linear(d_model, forward_expansion),
			nn.ReLU(),
			nn.Dropout(p),
			nn.Linear(forward_expansion, d_model)
		)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x):
		return self.block(x)


class BatchNorm(nn.Module):

	def __init__(self, d_model, eps=1e-6):
		super(BatchNorm, self).__init__()
		self.gamma = nn.Parameter(torch.ones(d_model))
		self.beta = nn.Parameter(torch.zeros(d_model))
		self.eps = eps
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x):
		normalized_output = self.gamma * (x - x.mean(dim=-1, keepdim=True)) / (
				x.std(dim=-1, keepdim=True) + self.eps) + self.beta
		return normalized_output


class EncoderLayer(nn.Module):

	def __init__(self, d_model, num_heads, p=0.1):
		super(EncoderLayer, self).__init__()
		self.batch_norm_1 = BatchNorm(d_model)
		self.batch_norm_2 = BatchNorm(d_model)
		self.attention = MultiHeadAttention(d_model, num_heads, p)
		self.block = FeedForwardBlock(d_model, p=p)
		self.dropout_1 = nn.Dropout(p)
		self.dropout_2 = nn.Dropout(p)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x, mask):
		x2 = self.batch_norm_1(x)
		x = x + self.dropout_1(self.attention(x2, x2, x2, mask=mask))
		x2 = self.batch_norm_2(x)
		x = x + self.dropout_2(self.block(x2))
		return x


class DecoderLayer(nn.Module):

	def __init__(self, d_model, num_heads, p=0.1):
		super(DecoderLayer, self).__init__()
		self.batch_norm_1 = BatchNorm(d_model)
		self.batch_norm_2 = BatchNorm(d_model)
		self.batch_norm_3 = BatchNorm(d_model)
		self.attention_1 = MultiHeadAttention(d_model, num_heads, p)
		self.attention_2 = MultiHeadAttention(d_model, num_heads, p)
		self.block = FeedForwardBlock(d_model, p=p)
		self.dropout_1 = nn.Dropout(p)
		self.dropout_2 = nn.Dropout(p)
		self.dropout_3 = nn.Dropout(p)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x, encoder_outputs, src_mask, tgt_mask):
		x2 = self.batch_norm_1(x)
		x = x + self.dropout_1(self.attention_1(x2, x2, x2, mask=tgt_mask))
		x2 = self.batch_norm_2(x)
		x = x + self.dropout_2(self.attention_2(x2, encoder_outputs, encoder_outputs, mask=src_mask))
		x2 = self.batch_norm_3(x)
		x = x + self.dropout_3(self.block(x2))
		return x


class Encoder(nn.Module):

	def __init__(self, vocab_size, d_model, num_heads, num_layers, p):
		super(Encoder, self).__init__()
		self.num_layers = num_layers
		self.src_embedding = nn.Embedding(vocab_size, d_model)
		self.positional_encoding = PositionalEncoding(d_model)
		self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, p) for _ in range(num_layers)])
		self.batch_norm = BatchNorm(d_model)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, src, mask):
		x = self.positional_encoding(self.src_embedding(src))
		for i in range(self.num_layers):
			x = self.encoder[i](x, mask)
		x = self.batch_norm(x)
		return x


class Decoder(nn.Module):

	def __init__(self, vocab_size, d_model, num_heads, num_layers, p):
		super(Decoder, self).__init__()
		self.num_layers = num_layers
		self.tgt_embedding = nn.Embedding(vocab_size, d_model)
		self.positional_encoding = PositionalEncoding(d_model)
		self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, p) for _ in range(num_layers)])
		self.batch_norm = BatchNorm(d_model)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, tgt, encoder_outputs, src_mask, tgt_mask):
		x = self.positional_encoding(self.tgt_embedding(tgt))
		for i in range(self.num_layers):
			x = self.decoder[i](x, encoder_outputs, src_mask, tgt_mask)
		x = self.batch_norm(x)
		return x


class Transformer(nn.Module):

	def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, num_layers, p):
		super(Transformer, self).__init__()
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.encoder = Encoder(len(src_vocab), d_model, num_heads, num_layers, p)
		self.decoder = Decoder(len(tgt_vocab), d_model, num_heads, num_layers, p)
		self.out = nn.Linear(d_model, len(tgt_vocab))
		self._init_parameters()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def _init_parameters(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	@staticmethod
	def create_pad_mask(matrix, pad_token):
		mask = (matrix.unsqueeze(1) != pad_token).type(torch.float32)
		mask = torch.bmm(mask.transpose(1, 2), mask)
		return mask

	def create_src_mask(self, src):
		return self.create_pad_mask(src, self.src_vocab['<pad>'])

	def create_tgt_mask(self, tgt):
		mask = self.create_pad_mask(tgt, self.tgt_vocab['<pad>'])
		mask = torch.tril(torch.ones(tgt.shape[1], tgt.shape[1])).to(self.device) * mask
		return mask

	def create_memory_mask(self, src, tgt):
		mask = self.create_pad_mask(src, self.src_vocab['<pad>'])
		if tgt.shape[1] == src.shape[1]:
			return mask
		elif tgt.shape[1] > src.shape[1]:
			return F.pad(mask, (0, 0, 0, tgt.shape[1] - src.shape[1]))
		else:
			return mask[:, :tgt.shape[1] - src.shape[1]]

	def forward(self, src, tgt):
		src_mask = self.create_src_mask(src)
		tgt_mask = self.create_tgt_mask(tgt)
		memory_mask = self.create_memory_mask(src, tgt)
		encoder_outputs = self.encoder(src, src_mask)
		decoder_outputs = self.decoder(tgt, encoder_outputs, memory_mask, tgt_mask)
		output = self.out(decoder_outputs)
		return output

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
