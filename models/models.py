
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import init
from torch.nn.modules.conv import Conv3d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_Encoder(nn.Module):
    '''
    CNN Network which encodes the voxelised ligands into a vectorised form 
    '''
    def __init__(self, in_channels=14) -> None:
        super().__init__()
        channels = in_channels
        layers = []
        # Define the VGG-16 network

        # 2 conv layers followed by max pooling

        # First block
        layers.append(nn.Conv3d(channels, 32, padding=1, kernel_size=3,stride=1))
        layers.append(nn.BatchNorm3d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(32, 32, padding=1,  kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool3d(stride=2, kernel_size=2))

        # Second block
        layers.append(nn.Conv3d(32, 64, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(64, 64, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(64))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool3d(stride=2, kernel_size=2))

        # Third block
        layers.append(nn.Conv3d(64, 128, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(128, 128, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(128, 128, padding=1, kernel_size=3, stride=1))
        layers.append(nn.BatchNorm3d(128))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool3d(stride=2, kernel_size=2))

#        layers.append(nn.Conv3d(128, 256, padding=1, kernel_size=3, stride=1))
#        layers.append(nn.BatchNorm3d(256))
#        layers.append(nn.ReLU())
#        layers.append(nn.Conv3d(256, 256, padding=1, kernel_size=3, stride=1))
#        layers.append(nn.BatchNorm3d(256))
#        layers.append(nn.ReLU())
#        layers.append(nn.Conv3d(256, 256, padding=1, kernel_size=3, stride=1))
#        layers.append(nn.BatchNorm3d(256))
#        layers.append(nn.ReLU())
#        layers.append(nn.MaxPool3d(stride=2, kernel_size=2))

        self.features = nn.Sequential(*layers)
#        self.fc = nn.Linear(128, 256)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.features(x)
        x1 = x1.permute(0, 2, 3, 4, 1)
#        x2 = x1.mean(dim=2).mean(dim=2).mean(dim=2)
        return x1

class AttentionBlock(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim) -> None:
        """
        Attention Network
        """
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, encoder_output, decoder_hidden_state):
        encoder_att = self.encoder_att(encoder_output)
        decoder_att = self.decoder_att(decoder_hidden_state)
        attention_add = encoder_att + decoder_att.unsqueeze(1)
        attention_add = self.relu(attention_add)
        full_att = self.full_att(attention_add).squeeze(2)
        alphas = self.softmax(full_att)
        att_weighted_encoding = (encoder_output * alphas.unsqueeze(2)).sum(dim=1)
        return att_weighted_encoding, alphas

class MolDecoder(nn.Module):
    def __init__(self, embed_size, decoder_dim, att_dim,
            vocab_size=29, dropout=0.5, encoder_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention_block = AttentionBlock(encoder_dim, decoder_dim, att_dim)
        self.embed_layer = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim,
                bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.final_layer = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed_layer.weight.data.uniform_(-0.1, 0.1)
        self.final_layer.weight.data.uniform_(-0.1, 0.1)
        self.final_layer.bias.data.fill_(0)

    def init_hidden_states(self, encoder_output):
        mean_encoder_output = encoder_output.mean(dim=1)
        h = self.init_h(mean_encoder_output)
        c = self.init_c(mean_encoder_output)
        return h,c

    def forward(self,encoder_output, captions, caption_lengths):
        batch_size = encoder_output.shape[0]
        encoder_dim = encoder_output.shape[-1]
        encoder_output = encoder_output.view(batch_size, -1, encoder_dim)
        total_voxels = encoder_output.shape[1]

        embeddings = self.embed_layer(captions)
        h,c = self.init_hidden_states(encoder_output)

        decode_lengths = caption_lengths

        predictions = torch.zeros(batch_size, max(decode_lengths),
                self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths),
                total_voxels).to(device)
        for t_i in range(max(decode_lengths)):
            eff_batch_size_t_i = sum([l > t_i for l in decode_lengths])
            att_weighted_encoding, alpha = self.attention_block(encoder_output[:eff_batch_size_t_i],
                    h[:eff_batch_size_t_i])
            gate = self.sigmoid(self.f_beta(h[:eff_batch_size_t_i]))
            att_weighted_encoding = gate * att_weighted_encoding
            h, c = self.lstm_cell( torch.cat([embeddings[:eff_batch_size_t_i,
                t_i, :], att_weighted_encoding], dim=1),
                (h[:eff_batch_size_t_i], c[:eff_batch_size_t_i]) )
            pred = self.final_layer(self.dropout(h))
            predictions[:eff_batch_size_t_i, t_i, :] = pred
            alphas[:eff_batch_size_t_i, t_i, :] = alpha
        return predictions, alphas, decode_lengths

# define weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)

if __name__ == "__main__":
    encoder_model = CNN_Encoder()
    rand_tensor = torch.randn(1, 14, 24,24, 24)
    encoder_output = encoder_model(rand_tensor)
    rand_tensor = torch.randn(4, 5, 24, 24, 24).to(device)
    cond_x = torch.randn(4, 3, 24, 24, 24).to(device)
    vae_decoder = VAE_Decoder().to(device)
    z = torch.randn(4, 256).to(device)
    recon_x = vae_decoder(z, cond_x)
    print(recon_x.shape)
    D = MultiDiscriminator().to(device)
    out = D(rand_tensor)
    for o in out:
        print(o.shape)
