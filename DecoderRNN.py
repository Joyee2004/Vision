import torch
import torch.nn as nn
from Attention import Attention

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim = 1536, decoder_dim= 512, drop_prob=0.3):
        super(DecoderRNN, self).__init__()

        # Save the model parameters
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeds = self.embedding(captions)
        h, c = self.init_hidden_state(features)

        seq_length = captions.size(1) - 1
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(features.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(features.device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))
            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def generate_caption(self, features, max_len=20, vocab=None):
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)

        alphas = []

        word = torch.tensor([vocab.stoi['<start>']]).to(features.device)
        embeds = self.embedding(word).unsqueeze(0)

        captions = []

        for _ in range(max_len):
            alpha, context = self.attention(features, h)
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds.squeeze(1), context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            predicted_word_idx = output.argmax(dim=1)

            captions.append(predicted_word_idx.item())

            if vocab.itos[predicted_word_idx.item()] == "<end>":
                break

            embeds = self.embedding(predicted_word_idx).unsqueeze(0)

        return [vocab.itos[idx] for idx in captions], alphas

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
