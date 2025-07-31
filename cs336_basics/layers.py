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
        d_ff: int,
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


class RoPE(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        """
        Bacha na k nevim, co je to za dtype -> mozna to killne celej performance idk
        """
        k = torch.arange(d_k // 2, dtype=torch.float32, device=device)
        self.theta = 1.0 / (theta ** (2 * k / d_k))

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Float[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Mame d * d matici, kde mám vícekrát rotační matici ve 2d
        Vstup:
            * obvykle vstup bude Query,Key
            * délka sekvence n, velikost batche b, každý token má e složek(velikost embeddingu), b n e po tranformaci na Query e d_q -> b n d_q
            * následně potřebujeme zrotovat všechny řádky všech matic pomocí d_q, d_q matice, ale je potřeba pamatovat, že pozice tokenu ve větě hraje roli
        Výstup
        """
        """
        Vypadá, že nechápu einsum, protože se automaticky neprovedla transpozice -> možná je to kvůli tomu, že matice je čtvercová
        """
        position_preped = rearrange(token_positions, " ... -> ... 1")
        angle = position_preped * self.theta
        R_cos = torch.cos(angle).unsqueeze(-1)
        R_sin = torch.sin(angle).unsqueeze(-1)
        R_final = rearrange(
            torch.stack([R_cos, -R_sin, R_sin, R_cos], dim=-1).squeeze(-2),
            " ... seq dk_2 (x y) -> ... seq dk_2 y x",
            x=2,
            y=2,
        )
        X_preped = rearrange(x, "... seq ( dk_2 y ) -> ... seq dk_2 1 y ", y=2)
        preres = einsum(
            X_preped,
            R_final,
            " ... seq dk_2 x1 y, ... seq dk_2 y x -> ... seq dk_2 x1 x",
        )
        res = rearrange(preres, "... seq dk_2 x1 x -> ... seq (dk_2 x1 x)")
        return res


def softmax(x: Float[torch.Tensor, "..."], dim: int):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


class Softmax(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return softmax(x, self.dim)


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... queries d_k"],
    K: Float[torch.Tensor, "... keys d_k"],
    V: Float[torch.Tensor, "... values d_v"],
    mask: Float[torch.Tensor, "queries keys"] | None = None,
):
    """
    Q = X W_q (seq_d_k)
    K = X W_k (seq d_k)
    ?softmax(Q K^T) -> mělibychom získat váhy pro každej vektor n * n
    """
    QK = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(
        K.size()[-1]
    )
    """
    Proč - nekonečno umožňuje maskovat chtělo by to prozkoumat gradient
    """
    if mask is not None:
        QK = torch.where(mask, QK, -torch.inf)
    soft = softmax(QK, dim=-1)
    attention = einsum(soft, V, "... queries values, ... values d_v -> ... queries d_v")
    return attention
