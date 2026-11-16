"""电脑端数据打包：把 L0 的量化矩阵转成 MVM 加速器的流格式。

不依赖 cocotb，上板（L3）时同样用它生成要写进 DDR / 经 XDMA 发送的字节。
"""
import numpy as np

N = 32          # tile 尺寸，与 rtl/mvm_top.sv 的参数 N 一致
SW_FRAC = 24
X_BASE = 0x1000
MAX_KB = 32     # x 缓冲 32 块 → K ≤ 1024

REG_CTRL, REG_STATUS, REG_KB, REG_NT, REG_SHIFT, REG_CYCLES, REG_ID = (
    0x000, 0x004, 0x008, 0x00C, 0x010, 0x014, 0x018)
MVM_ID = 0x4D564D01


def _ceil(a, b):
    return -(-a // b)


def pack_weights(wq, SW):
    """(M, K) int8 权重 + (M,) SW → (字节流, nt, kb)。

    每个输出 tile：N/8 拍 SW（uint32 小端），然后 kb 外层、行内层的 kb*N 拍权重。
    M、K 不是 N 的倍数时补零（补零行的 SW=0，输出 0，由驱动丢弃）。
    """
    wq = np.asarray(wq, dtype=np.int8)
    SW = np.asarray(SW, dtype=np.int64)
    M, K = wq.shape
    assert SW.shape == (M,) and SW.min() >= 0 and SW.max() < 2**31
    nt, kb = _ceil(M, N), _ceil(K, N)
    assert kb <= MAX_KB, f"K={K} 超出 x 缓冲（最多 {MAX_KB * N}）"
    W = np.zeros((nt * N, kb * N), dtype=np.int8)
    W[:M, :K] = wq
    S = np.zeros(nt * N, dtype="<u4")
    S[:M] = SW
    parts = []
    for t in range(nt):
        parts.append(S[t * N:(t + 1) * N].tobytes())
        tile = W[t * N:(t + 1) * N]                       # (N, kb*N)
        parts.append(tile.reshape(N, kb, N).transpose(1, 0, 2).tobytes())
    return b"".join(parts), nt, kb


class Packed:
    """预先打包好的权重矩阵（相当于常驻 DDR 的一段权重映像）。"""

    def __init__(self, wq, SW):
        self.M, self.K = wq.shape
        self.stream, self.nt, self.kb = pack_weights(wq, SW)


def pack_x(xq, kb):
    """int8 向量 → 32 位字列表（小端），写到 X_BASE 起的地址。"""
    x = np.zeros(kb * N, dtype=np.int8)
    xq = np.asarray(xq)
    assert np.abs(xq).max(initial=0) <= 127
    x[:len(xq)] = xq
    return np.frombuffer(x.tobytes(), dtype="<u4").tolist()


def shift_reg(s):
    sh = SW_FRAC - int(s)
    assert -128 <= sh <= 127
    return sh & 0xFF


def unpack_y(data, M):
    return np.frombuffer(bytes(data), dtype="<i4")[:M].astype(np.int64)


def ideal_cycles(M, K):
    """计算核的理论周期：每 tile = SW 拍 + kb*N 拍累加 + N 拍输出。"""
    nt, kb = _ceil(M, N), _ceil(K, N)
    return nt * (N // 8 + kb * N + N)
