import torch


class DDPM_Scheduler:

    def __init__(self, beta_min: float, beta_max: float, n_steps = 1000) -> None:

        """
        Beta is the strength of noise to be added at every timestep during Noisification.
        """

        self.n_timesteps = n_steps # 1000
        self.beta_min = beta_min # 0.0001
        self.beta_max = beta_max # 0.02
        self.beta = torch.linspace(
                                   self.beta_min, self.beta_max, self.n_timesteps, dtype=torch.float32
                                  )
        self.alpha = 1.0 - self.beta # Alpha = 1 - Beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # Alpha_hat = Cumulative product of Alpha.

    """_____________________________________________________________________________________________________________________________________________________________"""

    def diffusion_process(self, original_image: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        """
        Forward process or diffusion process, is fixed to a Markov chain that gradually adds 
        Gaussian noise to the data according to a variance schedule Beta_1, ......., Beta_T.

        Args:
        original_image -> Image to be noised.
        t -> timestep.
        noise -> Noise to be used in Reparameterization trick.

        Method:
        Noise is added to the image according to equation (4)[1]. 

        """
        sqrt_alpha_hat = (self.alpha_hat[t] ** 0.5).to(original_image.device)
        image_shape = original_image.size()
        sqrt_alpha_hat = sqrt_alpha_hat.reshape(original_image.size(0))

        while len(sqrt_alpha_hat.size()) < len(image_shape):
            sqrt_alpha_hat = sqrt_alpha_hat.unsqueeze(-1)

       # print(sqrt_alpha_hat.size())
        mean = sqrt_alpha_hat * original_image
        sigma = ((1 - self.alpha_hat[t]) ** 0.5).to(original_image.device)
        sigma = sigma.reshape(original_image.size(0))

        while len(sigma.size()) < len(image_shape):
            sigma = sigma.unsqueeze(-1)

        return mean + sigma * noise # Reparameterization trick
    
    """_____________________________________________________________________________________________________________________________________________________________"""

    def sample(self, xt: torch.Tensor, eta_theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
       
        """
        The sampling procedure, resembles Langevin dynamics with eta_theta as a learned gradient of the data density.

        Args:
        xt -> Pure noise.
        eta_theta -> Predicted noise.
        t -> timestep.
        
        Method:
        Mean - Parameterization using equation (11)[1].
        Variance - Beta
        
        """
        alpha = self.alpha[t].to(xt.device)
        alpha_hat = self.alpha_hat[t].to(xt.device)
        variance = self.beta[t].to(xt.device)
        sigma = variance ** 0.5

        mean = xt - ((1 - alpha) / (1 - alpha_hat) ** 0.5) * eta_theta
        mean = (1 / alpha ** 0.5) * mean

        if t == 0:
            z = 0
        else:
            z = torch.randn(xt.size(), device=xt.device)

        return mean + sigma * z # Reparameterization trick

"""_____________________________________________________________________________________________________________________________________________________________"""

"""
Reference:
1. https://arxiv.org/abs/2006.11239

"""