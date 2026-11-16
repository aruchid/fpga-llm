"""llama2.c 的 tokenizer.bin（SentencePiece BPE，按分数合并，字节回退）。"""
import struct


class Tokenizer:
    BOS, EOS = 1, 2

    def __init__(self, vocab, scores):
        self.vocab = vocab        # list[bytes]
        self.scores = scores      # list[float]
        self.lookup = {}
        for i, piece in enumerate(vocab):
            self.lookup.setdefault(piece, i)

    @classmethod
    def load(cls, path: str, vocab_size: int):
        vocab, scores = [], []
        with open(path, "rb") as f:
            f.read(4)  # max_token_length
            for _ in range(vocab_size):
                score, n = struct.unpack("<fi", f.read(8))
                vocab.append(f.read(n))
                scores.append(score)
        return cls(vocab, scores)

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        tokens = []
        if bos:
            tokens.append(self.BOS)
        if text:
            tokens.append(self.lookup[b" "])  # SentencePiece 的前导空格
        for ch in text:
            b = ch.encode("utf-8")
            if b in self.lookup:
                tokens.append(self.lookup[b])
            else:
                tokens.extend(byte + 3 for byte in b)  # 字节回退：<0x00> 从 3 开始
        start = 1 if bos else 0
        while True:  # 每次合并分数最高的相邻对
            best, best_id, best_i = -1e10, -1, -1
            for i in range(start, len(tokens) - 1):
                tid = self.lookup.get(self.vocab[tokens[i]] + self.vocab[tokens[i + 1]])
                if tid is not None and self.scores[tid] > best:
                    best, best_id, best_i = self.scores[tid], tid, i
            if best_i < 0:
                break
            tokens[best_i:best_i + 2] = [best_id]
        if eos:
            tokens.append(self.EOS)
        return tokens

    def decode_piece(self, prev: int, tok: int) -> bytes:
        piece = self.vocab[tok]
        if prev == self.BOS and piece.startswith(b" "):
            piece = piece[1:]
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            return bytes([int(piece[3:5], 16)])
        return piece

    def decode(self, tokens) -> str:
        out, prev = b"", self.BOS
        for t in tokens:
            if t == self.BOS:
                prev = t
                continue
            out += self.decode_piece(prev, t)
            prev = t
        return out.decode("utf-8", errors="replace")
