import torch
from einops import einsum
import math
from jaxtyping import Float


class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()

        self.W = torch.nn.Parameter(
            torch.zeros(size=(out_features, in_features), dtype=dtype, device=device)
        )
        """
        vstup:
        batchsize inf_eatures
        Váhy:
        out_features in features 
        """
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.W, mean=0, std=std, a=-3 * std, b=3 * std, generator=generator
        )

    def forward(self, x: Float[torch.Tensor, " ... d_in"]):
        return einsum(
            x,
            self.W,
            "... in_features, out_features in_features -> ... out_features",
        )


class Embeding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.Embeddings = torch.nn.Parameter(
            torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        torch.nn.init.trunc_normal_(self.Embeddings,mean=0,std=1,a=-3,b=3,generator=generator)
    
    def forward(self,x:Float[torch.Tensor,"..."]):
        """
        Mám (vocab_size embedding_dim)
        Vstup:
        ... sequence_length
        """
        return self.Embeddings[x]

