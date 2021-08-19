import numpy as np
from torch import optim
import torch.nn.functional as F
from datasets import english_german
from model import Transformer

train_loader, test_loader, vocab = english_german.load_data('../../data/english-german.csv')

epochs = 35
src_vocab, tgt_vocab = vocab['english'], vocab['german']
model = Transformer(src_vocab, tgt_vocab, 128, 8, 3, 0.1)
optimizer = optim.Adam(model.parameters())

src_sequence = next(iter(test_loader))[0]
for epoch in range(epochs):
    model.eval()
    print()
    print(src_vocab.reverse_tokenize(src_sequence))
    print(model.translate_sequence(src_sequence))
    print()
    model.train()
    losses = []
    for (source_batch, target_batch) in train_loader:
        output = model(source_batch, target_batch[:, :-1])
        output = output.reshape(-1, output.shape[-1])
        target = target_batch[:, 1:].reshape(-1)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target, ignore_index=tgt_vocab['<pad>'])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (len(losses) - 1) % 20 == 0:
            print(f'Epoch: {epoch}, Batch: {len(losses)}, Loss: {np.mean(losses)}')

