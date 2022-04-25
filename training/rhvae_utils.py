import torch

def create_metric(model):
    def G(z):
        #print(torch.exp(
            #-torch.norm(
                #model.centroids_tens.unsqueeze(0).to(z.device) - z.unsqueeze(1),dim=-1
                #)
                #** 2
                #/ (model.temperature**2)
            #).unsqueeze(-1).unsqueeze(-1))
        return torch.inverse(
            (
                model.M_tens.unsqueeze(0).to(z.device)
                * torch.exp(
                    -torch.norm(
                        model.centroids_tens.unsqueeze(0).to(z.device) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (model.temperature ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1)
            + model.lbd * torch.eye(model.model_config.latent_dim).to(z.device)
        )

    return G


def create_inverse_metric(model):
    def G_inv(z):
        #print(model.centroids_tens.unsqueeze(0).shape, z.unsqueeze(1).shape)
        #print((model.centroids_tens.unsqueeze(0).to(z.device) - z.unsqueeze(1)).shape)
        #print(torch.norm(model.centroids_tens.unsqueeze(0).to(z.device) - z.unsqueeze(1), dim=-1).shape)
        #print(torch.exp(
            #-torch.norm(model.centroids_tens.unsqueeze(0).to(z.device) - z.unsqueeze(1), dim=-1)
            #** 2
            #/ (model.temperature ** 2)
        #).shape)
        #print( 
            #(
            #model.M_tens.unsqueeze(0).to(z.device)
            #* torch.exp(
                #-torch.norm(model.centroids_tens.unsqueeze(0).to(z.device) - z.unsqueeze(1), dim=-1)
                #** 2
                #/ (model.temperature ** 2)
            #)
            #.unsqueeze(-1)
            #.unsqueeze(-1)
            #).sum(dim=1).shape
        #)
        return (
            model.M_tens.unsqueeze(0).to(z.device)
            * torch.exp(
                -torch.norm(model.centroids_tens.unsqueeze(0).to(z.device) - z.unsqueeze(1), dim=-1)
                ** 2
                / (model.temperature ** 2)
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1) + model.lbd * torch.eye(model.model_config.latent_dim).to(z.device)

    return G_inv
