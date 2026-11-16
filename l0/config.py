from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.dim * self.n_kv_heads // self.n_heads

    @property
    def kv_mul(self) -> int:
        return self.n_heads // self.n_kv_heads


# llama2.c 的 stories15M
STORIES15M = Config(dim=288, hidden_dim=768, n_layers=6, n_heads=6,
                    n_kv_heads=6, vocab_size=32000, seq_len=256)

# 单元测试用的小配置（带 GQA：4 个 query 头共享 2 个 KV 头）
TINY = Config(dim=64, hidden_dim=160, n_layers=2, n_heads=4,
              n_kv_heads=2, vocab_size=100, seq_len=32)
