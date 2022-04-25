class RH_VAE_CONFIG:
    def __init__(self, is_train = True, input_dim=(5,24,24,24),\
            num_channels=5, latent_dim=10, n_lf_steps=10, eps_lf=0.001,\
            beta_zero=0.3, temperature=1.5, regularization=0.01, reconstruction_loss="bce"):
        self.is_train: bool = is_train
        self.input_dim = input_dim
        self.num_channels: int = num_channels
        self.latent_dim: int = latent_dim
        self.n_lf_steps: int = n_lf_steps
        self.eps_lf: float = eps_lf
        self.beta_zero: float = beta_zero
        self.temperature: float = temperature
        self.regularization: float = regularization
        self.reconstruction_loss: str = reconstruction_loss
