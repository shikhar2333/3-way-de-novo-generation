from .rhvae_sampler_config import RHVAESamplerConfig
from .riemann_tools import Exponential_map
import torch
#stores_PE = open("scripts/PE.txt", "w")

class RHVAE_Sampler:
    def __init__(self, model, cond_x=None, n_samples=1, mcmc_steps=10, \
            sampling_method="HMC") -> None:
        sampler_config = RHVAESamplerConfig(num_samples=n_samples,
                mcmc_steps=mcmc_steps)
        self.sampling_method = sampling_method
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.centroids_tens = self.model.centroids_tens.to(self.device)
        self.model.M_tens = self.model.M_tens.to(self.device)

        self.num_samples = sampler_config.num_samples
        self.batch_size = sampler_config.batch_size
        self.mcmc_steps_nbr = sampler_config.mcmc_steps
        self.n_lf = torch.tensor([sampler_config.n_lf]).to(self.device)
        self.eps_lf = torch.tensor([sampler_config.eps_lf]).to(self.device)
        self.beta_zero_sqrt = (
            torch.tensor([sampler_config.beta_zero]).to(self.device).sqrt()
        )
        if cond_x != None:
            self.cond_x = cond_x.to(self.device)
        else:
            self.cond_x = None

    def sample(self):
        complete_examples = int(self.num_samples/self.batch_size)
        remaining_examples = self.num_samples%self.batch_size

        ligand_shapes_gen = []
        z_gen = []
        cond_x = self.cond_x.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)

        if self.sampling_method == "HMC":

            for i in range(complete_examples):
                samples = self.hmc_sampling(self.batch_size)
                if self.cond_x != None:
                    x_gen = self.model.decoder(samples,cond_x).detach()
                    ligand_shapes_gen.append(x_gen)
                else:
                    z_gen.append(samples)

            if remaining_examples > 0:
                cond_x = self.cond_x.unsqueeze(0).repeat(remaining_examples, 1, 1, 1, 1)
                samples = self.hmc_sampling(remaining_examples)
                if self.cond_x != None:
                    x_gen = self.model.decoder(samples,cond_x).detach()
                    ligand_shapes_gen.append(x_gen)
                else:
                    z_gen.append(samples)

        elif self.sampling_method == "RW":
            for i in range(complete_examples):
                samples = self.random_walk_batch(n_samples=self.batch_size)
                if self.cond_x != None:
                    x_gen = self.model.decoder(samples,cond_x).detach()
                    ligand_shapes_gen.append(x_gen)
                else:
                    z_gen.append(samples)

            if remaining_examples > 0:
                cond_x = self.cond_x.unsqueeze(0).repeat(remaining_examples, 1, 1, 1, 1)
                samples = self.random_walk_batch(n_samples=remaining_examples)
                if self.cond_x != None:
                    x_gen = self.model.decoder(samples,cond_x).detach()
                    ligand_shapes_gen.append(x_gen)
                else:
                    z_gen.append(samples)
        elif self.sampling_method == "prior":
            for i in range(complete_examples):
                samples = torch.randn(self.batch_size,
                        self.model.model_config.latent_dim).to(self.device)
                if self.cond_x != None:
                    x_gen = self.model.decoder(samples,cond_x).detach()
                    ligand_shapes_gen.append(x_gen)
                else:
                    z_gen.append(samples)

            if remaining_examples > 0:
                cond_x = self.cond_x.unsqueeze(0).repeat(remaining_examples, 1, 1, 1, 1)
                samples = torch.randn(remaining_examples,
                        self.model.model_config.latent_dim).to(self.device)
                if self.cond_x != None:
                    x_gen = self.model.decoder(samples,cond_x).detach()
                    ligand_shapes_gen.append(x_gen)
                else:
                    z_gen.append(samples)
        return torch.cat(ligand_shapes_gen, dim=0) if self.cond_x!=None else \
                    torch.cat(z_gen, dim=0)

    def random_walk_batch(
        self,
        z0=None,
        latent_dim=10,
        n_steps=10,
        n_samples=1,
        delta=1.0,
        dt=1,
        verbose=False,
    ):
    
        device = self.device    
        with torch.no_grad():
            exponential_map = Exponential_map(latent_dim=latent_dim, device=device)
            acc_nbr = torch.zeros((n_samples, 1)).to(device)
            cov = torch.eye(latent_dim) * delta ** 2 * dt
            norm = torch.distributions.MultivariateNormal(
                loc=torch.zeros(latent_dim), covariance_matrix=cov
            )
    
            if z0 is None:
                idx = torch.randint(len(self.model.centroids_tens), (n_samples,))
                z0 = self.model.centroids_tens[idx]
                
            z = z0
            for i in range(n_steps):
    
                # Sample Velocities
                v = norm.sample((n_samples,))
    
                # Shoot
                z_traj, q_traj = exponential_map.shoot(p=z, v=v, model=self.model, n_steps=10)
                z = z_traj[:, -1, :].to(device)
    
                # Compute acceptance ratio
                alpha = (
                    torch.det((self.model.G_inv(z))).sqrt() / torch.det((self.model.G_inv(z0))).sqrt()
                )
                acc = torch.rand(n_samples).to(device)
                #moves = torch.tensor(acc < alpha).type(torch.int).reshape(n_samples, 1)
                moves = (acc < alpha).type(torch.int).reshape(n_samples,
                        1).clone().detach()
                z = z * moves + (1 - moves) * z0
                acc_nbr += moves
    
                z0 = z
    
                if i % 100 == 0 and verbose:
                    if i == 0:
                        print(f"Iteration {i} / {n_steps}")
                    else:
                        print(
                            f"Iteration {i} / {n_steps}\t Mean acc. rate {torch.mean(100*(acc_nbr / (i+1)))}"
                        )
        return z
    
    def hmc_sampling(self, n_samples: int):
        with torch.no_grad():

            idx = torch.randint(len(self.model.centroids_tens), (n_samples,))

            z0 = self.model.centroids_tens[idx]

            beta_sqrt_old = self.beta_zero_sqrt
            z = z0
            for i in range(self.mcmc_steps_nbr):

                gamma = torch.randn_like(z, device=self.device)
                rho = gamma / self.beta_zero_sqrt

                H0 = -self.log_sqrt_det_G_inv(z, self.model) + 0.5 * torch.norm(rho, dim=1) ** 2
                #print(-self.log_sqrt_det_G_inv(z, self.model),
                        #0.5*torch.norm(rho, dim=1)**2)
                #print("init energy: ",H0)
                #print(self.model.G(z).det())

                for k in range(self.n_lf):

                    g = -self.grad_log_prop(z, self.model).reshape(
                        n_samples, self.model.model_config.latent_dim
                    )
                    # step 1
                    rho_ = rho - (self.eps_lf / 2) * g

                    # step 2
                    z = z + self.eps_lf * rho_

                    g = -self.grad_log_prop(z, self.model).reshape(
                        n_samples, self.model.model_config.latent_dim
                    )
                    # g = (Sigma_inv @ (z - mu).T).reshape(n_samples, 2)

                    # step 3
                    rho__ = rho_ - (self.eps_lf / 2) * g

                    # tempering
                    beta_sqrt = self.tempering(
                        k + 1, self.n_lf, self.beta_zero_sqrt
                    )

                    rho = (beta_sqrt_old / beta_sqrt) * rho__
                    beta_sqrt_old = beta_sqrt

                H = -self.log_sqrt_det_G_inv(z, self.model) + 0.5 * torch.norm(rho, dim=1) ** 2
                #stores_PE.write(str(-self.log_sqrt_det_G_inv(z,
                    #self.model)[0].item()) +
                        #'\n')
                #print(H)
                #print(torch.exp(-H),torch.exp(-H0))
                alpha = torch.exp(-H) / (torch.exp(-H0))
                acc = torch.rand(n_samples).to(self.device)
                #print("Acceptance probability and Sampled probability: ",alpha, acc)
                moves = (acc < alpha).type(torch.int).reshape(n_samples, 1)
                z = z * moves + (1 - moves) * z0
                #print("PE and KE after acceptance/rejection: ", -self.log_sqrt_det_G_inv(z, self.model),
                        #0.5*torch.norm(rho, dim=1)**2)

                z0 = z

            #stores_PE.close()
            return z

    
    def tempering(self, k, K, beta_zero_sqrt):
        beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt

        return 1 / beta_k
    
    def log_sqrt_det_G_inv(self, z, model):
        #print(model.M_tens.shape)
        #print(torch.logdet(model.G_inv(z)))
        #return 0.5*torch.logdet(model.G_inv(z))
        return torch.log(torch.sqrt(torch.det(model.G_inv(z))) + 1e-10)

    def grad_log_sqrt_det_G_inv(self, z, model):
        return (
            -0.5
            * torch.transpose(model.G(z), 1, 2)
            @ torch.transpose(
                (
                    -2
                    / (model.temperature ** 2)
                    * (model.centroids_tens.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(2)
                    @ (
                        model.M_tens.unsqueeze(0)
                        * torch.exp(
                            -torch.norm(
                                model.centroids_tens.unsqueeze(0) - z.unsqueeze(1),
                                dim=-1,
                            )
                            ** 2
                            / (model.temperature ** 2)
                        )
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                ).sum(dim=1),
                1,
                2,
            )
        )

    def grad_log_prop(self, z, model):
        return self.grad_log_sqrt_det_G_inv(z, model)
