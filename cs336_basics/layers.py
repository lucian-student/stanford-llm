import torch
from einops import einsum, reduce, rearrange
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
        torch.nn.init.trunc_normal_(
            self.Embeddings, mean=0, std=1, a=-3, b=3, generator=generator
        )

    def forward(self, x: Float[torch.Tensor, "..."]):
        """
        Mám (vocab_size embedding_dim)
        Vstup:
        ... sequence_length
        """
        return self.Embeddings[x]


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.nn.Parameter(
            torch.ones(size=(self.d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: Float[torch.Tensor, "... d_model"]):
        dtype = x.dtype
        casted = x.to(torch.float32)
        mean_squared = (
            (1 / self.d_model) * reduce(casted**2, "... d_model -> ...", "sum")
        ) + self.eps
        res = casted * self.gain
        res = res / rearrange(torch.sqrt(mean_squared), "... -> ... 1")
        return res.to(dtype)


class SiLU(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Float[torch.Tensor, "..."]):
        return torch.nn.functional.sigmoid(x) * x


class SwiGLU(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff:int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.W1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.W3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # ... d_ff
        self.W2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.silu = SiLU()

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))
