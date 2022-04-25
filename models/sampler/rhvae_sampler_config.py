class RHVAESamplerConfig:
    def __init__(self, num_samples=1, batch_size=16, mcmc_steps=10,\
            n_lf=15, eps_lf=0.03, beta_zero=1.0):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.mcmc_steps = mcmc_steps
        self.n_lf = n_lf
        self.eps_lf = eps_lf
        self.beta_zero = beta_zero
