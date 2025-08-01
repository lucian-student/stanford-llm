import torch
from einops import einsum, reduce, rearrange
import math
from jaxtyping import Float, Int


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
        """
        Důvod, proč se využívá transpozice, je protože potom se násobí řádek s řádkem, tím pádem lépe využíváme cache.
        """

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

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        dtype: torch.dtype | None = None,
        device=None,
    ):
        super().__init__()
        """
        Bacha na k nevim, co je to za dtype -> mozna to killne celej performance idk
        """
        k = torch.arange(d_k // 2, dtype=dtype, device=device)
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
    Můj odhad, že softmax vytvoří nulu, proto tam nepůjde gradient
    """
    if mask is not None:
        QK = torch.where(mask, QK, -torch.inf)
    soft = softmax(QK, dim=-1)
    attention = einsum(soft, V, "... queries values, ... values d_v -> ... queries d_v")
    return attention


def batch_cartesian_prod(
    a: Int[torch.Tensor, "... seq"], b: Int[torch.Tensor, "... seq"]
):
    seq = a.shape[-1]
    cols = b.repeat(*[1 for _ in range(len(b.shape) - 1)], seq)
    rows = a.unsqueeze(-1)
    rows = rearrange(
        rows.repeat(*[1 for _ in range(len(rows.shape) - 1)], seq),
        "... seq1 seq2 -> ... (seq1 seq2)",
        seq2=seq,
    )
    combined = torch.stack([rows.unsqueeze(-1), cols.unsqueeze(-1)], dim=-1).squeeze(-2)
    mask = rearrange(
        combined, "... (seq1 seq2) coords -> ... seq1 seq2 coords", seq2=seq
    )
    mask = mask[..., 0] >= mask[..., 1]
    return mask


class MultiheadAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        # use_rope: bool = True,
        theta: float = 10000,
        max_seq_len: int = 200,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        """
        Basicaly máme num_heads matic W_q(emb,d_k),W_k(emb,d_k),W_v(emb,d_v) -> údajně se da je všechny zkombinovat do jedné matice
        * základní postup máme jednu matici W_q pro všechny hlavy
        * optimalnější postup W_q,W_k,W_v jsou všechny v jedné matici
        Máme dále matici W_o(num_heads*d_v,emb), která slouží pro zkombinování více hlav
        emb je asi d_in
        
        mam W_q matici jak vytovřim všechny Q matice najednou
        chci asi matici Q(n,h * d_k)
        X(n,emb) W_q(emb,h * d_k) = Q(n,h*d_k)
        X(n,emb) W_k(emb,h * d_k) = K(n,h*d_k)
        X(n,emb) W_v(emb,h * d_v) = V(n,h*d_v)
        Combined(emb,2 * h * d_k + h * d_v) = Comb(n,2 * h * d_k + h * d_v)
        Potom bude potřeba správně slicovat

        Causal Masking(Nejdřív musím pochopit, co je kauzal masking):

        Důvod je, že při predikci dalšího tokenu bychom neměli vědět o dalších tokenech.
        * první token je ovlivněn pouze sám sebou
        * druhý token je ovlivněn sebou a předchozím tokenem
        Takže musím zkonstruovat masku, která je nad vedlejší diagonálou
        
        Aplikace Rope: Rope je aplikované na Query a Key, prý head dimenze má byt batch dimenzí, takže asi nějaký rearrange
        """
        # self.use_rope = use_rope
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_combined = Linear(
            in_features=d_model,
            out_features=3 * num_heads * self.d_k,
            dtype=dtype,
            device=device,
        )
        self.W_output = Linear(
            in_features=self.d_k * self.num_heads,
            out_features=d_model,
            dtype=dtype,
            device=device,
        )
        self.rope = RoPE(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq d_in"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, "... seq d_in"]:
        """
        Nevim jestli náhodou se může lišit délka sekvence během běhu modelu

        d_in = d_model
        """
        device = x.device
        dtype = x.dtype
        combined: Float[torch.Tensor, "... seq all"] = self.W_combined(x)
        # teď je potřeba aplikovat ROPE na query a key
        combined = rearrange(
            combined,
            " ... seq (matrices heads d_k) -> ... matrices heads seq d_k",
            matrices=3,
            heads=self.num_heads,
        )
        matrices = torch.split(combined, 1, dim=-4)
        Q, K, V = [m.squeeze(-4) for m in matrices]
        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
            mask = batch_cartesian_prod(token_positions, token_positions)
        else:
            seq = Q.shape[-2]
            positions = torch.arange(seq, dtype=dtype, device=device)
            mask = batch_cartesian_prod(positions, positions)
        attention = scaled_dot_product_attention(Q, K, V, mask)
        # může být chyba v tom jak jsem to zapojil možná prohodit d_v a heads???? chtělo by to si dát repete ohledně einsum notace
        concated_attention = rearrange(
            attention, "... heads seq d_v -> ... seq (heads d_v)"
        )
        multihead_attention = self.W_output(concated_attention)
        return multihead_attention


class TranformerBlock(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000,
        max_seq_len: int = 200,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.multihead_attention = MultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model=d_model, d_ff=d_ff, dtype=dtype, device=device)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "... seq d_model"]
    ) -> Float[torch.Tensor, "... seq d_model"]:
        """
        Pozor, nevím jestli se nemají pozice nějak měnit, nebo můžou být takhle fixně
        """
        dtype = x.dtype
        device = x.device
        seq = x.shape[-2]
        positions = torch.arange(seq, dtype=dtype, device=device)
        x = x + self.multihead_attention(self.norm1(x), positions)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = Embeding(
            num_embeddings=vocab_size, embedding_dim=d_model, dtype=dtype, device=device
        )
        self.blocks = torch.nn.ModuleList(
            [
                TranformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=rope_theta,
                    max_seq_len=context_length,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype
        )

    def forward(
        self, x: Int[torch.Tensor, "... seq"]
    ) -> Float[torch.Tensor, "... seq vocab_size"]:
        x = self.embedding(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.norm(x)
        x = self.output_embedding(x)
        return x
