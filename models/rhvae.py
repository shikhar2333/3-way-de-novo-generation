import torch
import os
from copy import deepcopy
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
#from rhconfig import RH_VAE_CONFIG
from collections import deque

class RH_VAE_CONFIG:
    def __init__(self):
        self.is_train: bool = True
        #self.is_train: bool = False
        self.input_dim = (5, 24, 24, 24)
        self.num_channels: int = 5
        self.latent_dim = 10
        self.n_lf_steps: int = 3
        self.eps_lf: float = 0.001
        self.beta_zero: float = 0.3
        self.temperature: float = 1.5
        self.regularization: float = 0.01
        self.reconstruction_loss: str = "bce"

config = RH_VAE_CONFIG()

def reparametrize(mu, logvar):
   std = logvar.mul(0.5).exp_()
   # sample from N(0, I)
   eps = torch.randn(*mu.size()).to('cuda')
   return mu + std * eps, eps
########################
# AutoEncoder Networks #
########################
def Conv_Block_3D(in_channels, out_channels, kernel_size, stride,
        padding, activation="lrelu", normalize=True, transpose=False, eps=1e-5):
    layers = []
    if not transpose:
        layers.append(nn.Conv3d(in_channels, out_channels, \
                kernel_size, stride, padding))
    else:
        layers.append(nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride, padding))

    if normalize:
        layers.append(nn.BatchNorm3d(out_channels, eps=eps))
    if activation == "lrelu":
        layers.append(nn.LeakyReLU(0.2))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    else:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

# Encoder Network for VAE
class Encoder(nn.Module):
    def __init__(self, num_channels=5, latent_dim=512):
        super().__init__()
        self.e1 = Conv_Block_3D(num_channels, 32, 4, 2, 1)

        self.e2 = Conv_Block_3D(32, 64, 4, 2, 1)

        self.e3 = Conv_Block_3D(64, 128, 4, 2, 1)

        self.e4 = Conv_Block_3D(128, latent_dim, 3, 1, 1)

        #self.e4 = Conv_Block_3D(64, latent_dim, 4, 2, 1)
        #self.e5 = Conv_Block_3D(latent_dim, latent_dim, 4, 2, 1)

        self.fc1 = nn.Linear(latent_dim * 3 * 3 * 3, latent_dim)
        self.fc2 = nn.Linear(latent_dim * 3 * 3 * 3, latent_dim)
        # mu and logvar
        self.latent_dim = latent_dim
    
    def forward(self, x):
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        #x = self.e5(x)
        x = x.view(-1, self.latent_dim*3*3*3)
        return self.fc1(x), self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, num_channels=5, latent_dim=512):
        super().__init__()
        self.d1 = nn.Linear(latent_dim, latent_dim*3*3*3)
        self.d2 = Conv_Block_3D(latent_dim, 128, 4, 2, 1,\
                  transpose=True, eps=1e-3)
        self.d3 = Conv_Block_3D(128, 64, 4, 2, 1,\
                  transpose=True, eps=1e-3)
        self.d4 = Conv_Block_3D(64 + 32, 64, 4, 2, 1,\
                  transpose=True, eps=1e-3)
        #self.d4 = Conv_Block_3D(64, 64, 4, 2, 1,\
                  #transpose=True, eps=1e-3)
        #self.d5 = Conv_Block_3D(latent_dim//4 + 32, latent_dim//16, 4, 2, 1,\
                  #transpose=True, eps=1e-3)
        #self.d5 = Conv_Block_3D(latent_dim//4 , latent_dim//8, 4, 2, 1,\
                  #transpose=True, eps=1e-3)
        self.d5 = nn.Conv3d(64 + 32, num_channels, 3, 1, 1)
        #self.d5 = nn.Conv3d(64, num_channels, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.latent_dim = latent_dim
        # Condtional encoding
        self.ce1 = nn.Conv3d(3, 32, 3, 1, 1)
        self.ce2 = nn.Conv3d(32, 32, 3, 2, 1)

    def forward(self, z, cond_x=None):
        # Conditional block
        cc1 = self.relu(self.ce1(cond_x))
        cc2 = self.relu(self.ce2(cc1))
        z = self.d1(z)
        z = self.relu(z)
        z = z.view(-1, self.latent_dim, 3, 3, 3)
        z = self.d2(z)
        z = self.d3(z)
        z = torch.cat([z, cc2], dim=1)
        z = self.d4(z)
        z = torch.cat([z, cc1], dim=1)
        #z = self.d5(z)
        #z = torch.cat([z, cc1], dim=1)
        return self.sigmoid(self.d5(z))

class Metric_MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
#
        self.layers = nn.Sequential(nn.Linear(np.prod(args.input_dim), 400), nn.ReLU())
        #self.layers = nn.Sequential(nn.Linear(np.prod(args.input_dim), 40), nn.ReLU())
        self.diag = nn.Linear(400, self.latent_dim)
        #self.diag = nn.Linear(40, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(400, k)
        #self.lower = nn.Linear(40, k)

    def forward(self, x):

        h1 = self.layers(x.reshape(-1, np.prod(self.input_dim)))
        h21, h22 = self.diag(h1), self.lower(h1)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
        indices = torch.tril_indices(
            row=self.latent_dim, col=self.latent_dim, offset=-1
        )

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h22

        # add diagonal coefficients
        L = L + torch.diag_embed(h21.exp())
        return L

class RH_VAE(nn.Module):
    def __init__(self, model_config = config):
        super().__init__()
        self.model_config = model_config
        self.n_lf_steps = model_config.n_lf_steps
        self.encoder = Encoder(num_channels=model_config.num_channels,
                latent_dim=model_config.latent_dim)
        self.decoder = Decoder(num_channels=model_config.num_channels,\
                latent_dim = model_config.latent_dim)

        self.metric = Metric_MLP(model_config)
        self.temperature = nn.Parameter(
                torch.tensor([model_config.temperature]), requires_grad=False 
        )
        self.lbd = nn.Parameter(
                torch.tensor([model_config.regularization], requires_grad=False)
        )
        self.beta_zero_sqrt = nn.Parameter(
            torch.tensor([model_config.beta_zero]), requires_grad=False
        )
        self.n_lf_steps = model_config.n_lf_steps
        self.eps_lf = model_config.eps_lf

        # this is used to store the matrices and centroids throughout training for
        # further use in metric update (L is the cholesky decomposition of M)
        self.M = deque(maxlen=100)
        self.centroids = deque(maxlen=100)

        self.M_tens = torch.randn(
            1, model_config.latent_dim, model_config.latent_dim
        )
        self.centroids_tens = torch.randn(1, model_config.latent_dim)

        # define a starting metric (c_i = 0 & L = I_d)
        def G(z):
            return torch.inverse(
                (
                    torch.eye(model_config.latent_dim, device=z.device).unsqueeze(0)
                    * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(model_config.latent_dim).to(z.device)
            )

        def G_inv(z):
            return (
                torch.eye(model_config.latent_dim, device=z.device).unsqueeze(0)
                * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(model_config.latent_dim).to(z.device)

        self.G = G
        self.G_inv = G_inv

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # sample from N(0, I)
        eps = torch.randn(*mu.size()).to(mu.device)
        return mu + std * eps, eps

    def _log_p_x_given_z(self, recon_x, x):
        # maximum likelihood of p(x|z) is approximately the reconstruction
        # error between recon_x, x
        if self.model_config.reconstruction_loss == "mse":
            # sigma is taken as I_D
            recon_loss = -0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1) 
            - torch.log(torch.tensor([2 * np.pi]).to(x.device)) \
                * np.prod(self.input_dim) / 2

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = -F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        else:
            return 0

        return recon_loss

    def _log_p_z(self, z):
        """
        Return Normal density function as prior on z
        """
        # define a N(0, I) distribution
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.model_config.latent_dim).to(z.device),
            covariance_matrix=torch.eye(self.model_config.latent_dim).to(z.device),
        )
        return normal.log_prob(z)

    def _log_p_xz(self, recon_x, x, z):
        log_p_x_given_z = self._log_p_x_given_z(recon_x, x)
        log_p_z = self._log_p_z(z)
        return log_p_x_given_z + log_p_z

    def _hamiltonian(self, recon_x, x, z, rho, G_inv, G_log_det):
        """
        H_x(z, rho) = -log(P(x,z,rho)) = -log(P(x,z)) -log(P(rho)) = U_x(z) + \
                      0,5*log((2*pi)^d*det(G(z))) + 0.5*rho^T*G(z)^-1*rho
        P(rho) = exp(-rho^T*G(z)^-1*rho)/sqrt( (2*pi)^d*det(G(z)) )
        """
        norm = (
            torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)
        ).sum()

        return -self._log_p_xz(recon_x, x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()
    
    def _leap_step_1(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        def next_rho(rho_prime):
            H = self._hamiltonian(recon_x, x, z, rho_prime, G_inv, G_log_det)
            grad_H_z = torch.autograd.grad(H, z, retain_graph=True)[0]
            return rho - 0.5*self.eps_lf*grad_H_z

        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = next_rho(rho_)
        return rho_

    def _leap_step_2(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        H_init = self._hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grad_H_z_init = torch.autograd.grad(H_init, z, retain_graph=True)[0]
        def next_z(z_prime):
            H = self._hamiltonian(recon_x, x, z_prime, rho, G_inv, G_log_det)
            grad_H_z = torch.autograd.grad(H, z, retain_graph=True)[0]
            return z + 0.5*self.eps_lf*(grad_H_z_init + grad_H_z)

        z_ = z.clone()
        for _ in range(steps):
            z_ = next_z(z_)
        return z_

    def _leap_step_3(self, recon_x, x, z, rho, G_inv, G_log_det):
        """
        Resolves third equation of generalized leapfrog integrator
        """
        H = self._hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grad_H_z = torch.autograd.grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * grad_H_z

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    def compute_loss(self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var,
            G_inv, G_log_det):
        logpxz = self._log_p_xz(recon_x, x, zK)  # log p(x, z_K)
        logrhoK = (
            (
                -0.5
                * (
                    torch.transpose(rhoK.unsqueeze(-1), 1, 2)
                    @ G_inv
                    @ rhoK.unsqueeze(-1)
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det
            )
            #- torch.log(torch.tensor([2 * np.pi]).to(x.device)) * self.latent_dim / 2
        )  # log p(\rho_K)

        logp = logpxz + logrhoK

        # define a N(0, I) distribution
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.model_config.latent_dim).to(x.device),
            covariance_matrix=torch.eye(self.model_config.latent_dim).to(x.device),
        )

        logq = normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # log(q(z_0|x))

        return -(logp - logq).mean(dim=0)

    def forward(self, x, cond_x=None):
        mu, logvar = self.encoder(x)
        z_init, eps0 = self.reparametrize(mu, logvar)
        z = z_init
        if self.model_config.is_train:
            # update the metric using batch data points
            L = self.metric(x)

            M = L @ torch.transpose(L, 1, 2)

            # store LL^T and mu(x_i) to update final metric
            self.M.append(M.detach().clone())
            self.centroids.append(mu.detach().clone())

            G_inv = (
                M.unsqueeze(0)
                * torch.exp(
                    -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                    / (self.temperature ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.model_config.latent_dim).to(x.device)
        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = np.linalg.cholesky(G.detach().cpu().numpy())
            L = torch.from_numpy(L).to(z.device)
            #L = torch.linalg.cholesky(G)
            #M = L @ torch.transpose(L, 1, 2)

        G_log_det = -torch.logdet(G_inv)
        #print(G_log_det)
        # sample initial velocity from N(0,I)
        velocity = torch.randn_like(z, device=x.device)
        rho = velocity / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        # sample \rho from N(0, G)
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decoder(z, cond_x)

        for k in range(self.model_config.n_lf_steps):

            # perform leapfrog steps

            # step 1
            rho_ = self._leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)

            # step 2
            z = self._leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)

            recon_x = self.decoder(z, cond_x)
            if self.model_config.is_train:
                G_inv = (
                    M.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                        / (self.temperature ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1) + self.lbd * torch.eye(self.model_config.latent_dim).to(x.device)
                torch.eye(self.model_config.latent_dim).to(x.device)

            else:
                # compute metric value on new z using final metric
                G = self.G(z)
                G_inv = self.G_inv(z)

            G_log_det = -torch.logdet(G_inv)
            #print(torch.det(G_inv))
            #print(torch.det(G))

            # step 3
            rho__ = self._leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf_steps)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        loss = self.compute_loss(recon_x, x, z_init, z, rho, eps0, velocity,
                mu, logvar, G_inv, G_log_det)
        return recon_x, z, loss

    def _update_metric(self):
        # convert to 1 big tensor
        self.M_tens = torch.cat(list(self.M))
        self.centroids_tens = torch.cat(list(self.centroids))

        # define new metric
        def G_inv(z):
            return (
                self.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (self.temperature ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.model_config.latent_dim).to(z.device)

        def G(z):
            return torch.inverse(
                (
                    self.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (self.temperature ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(self.model_config.latent_dim).to(z.device)
            )
            #return torch.inverse(G_inv(z))

        self.G = G
        self.G_inv = G_inv
        self.M = deque(maxlen=100)
        self.centroids = deque(maxlen=100)

    def save(self, epoch, dir_path: str = "/scratch/shubham/saved_models"):
        """Method to save the model at a specific location
        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """
        model_path = dir_path

        model_dict = {
            "M": deepcopy(self.M_tens.clone().detach()),
            "centroids": deepcopy(self.centroids_tens.clone().detach()),
            "model_state_dict": deepcopy(self.state_dict()),
        }
        os.makedirs(dir_path, exist_ok=True)
        model_file = "rh_vae_" + str(epoch) + ".pt"

        torch.save(model_dict, os.path.join(model_path, model_file))

    def load_model(self, model_file: str):
        model_weights = torch.load(model_file, map_location="cpu")
        return model_weights["model_state_dict"], model_weights["M"],\
                model_weights["centroids"]

