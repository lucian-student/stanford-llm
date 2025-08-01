import torch
from jaxtyping import Float, Int

class CELosss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: Float[torch.Tensor, "... seq vocab_size"],
        y: Int[torch.Tensor, "... seq"],
    ) -> Float[torch.Tensor, "..."]:
        """
        1. Odečtu maximum -> pro stabilitu jako u softmaxu
        2. potom budeš v podstatě počítat log(softmax)
            * čitatel je log(exp) - což se pokrátí
            * jmenovatel je log(sum(exp)) -> asi se nic nepokrátí
        3. nakonec mean(-log(softmax))
        """
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_stabilized = x - x_max
        numerator = torch.gather(x_stabilized,dim=-1,index=y.unsqueeze(-1)).squeeze(-1)
        denominator = torch.log(torch.sum(torch.exp(x_stabilized), dim=-1))
        return torch.mean(denominator - numerator, dim=-1)
