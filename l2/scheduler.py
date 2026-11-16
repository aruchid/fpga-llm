"""电脑端调度器：按 L0 硬件等价模型的运算流推理，运算交给 RTL 执行。

- 模型流程只有一份（l0.ref_hw.HWModel.forward_steps），这里只负责执行它 yield 出来的运算
- 共享同一输入的请求（q/k/v、w1/w3）合并成一个大矩阵，一次任务算完，减少启动和写 x 的次数。
  各输出行互不影响，所以合并不改变任何一位结果
- 权重在初始化时一次性打包（相当于上板时预先加载到 DDR）
- 非线性运算可选：交给 RTL 向量单元（vpu），或在电脑端用 L0 的定点代码计算（不占仿真时间）
"""
import time

import numpy as np

from l0.ref_hw import HWModel, MatmulReq, exec_op
from l0.tokenizer import Tokenizer
from l2 import pack as pk


class RTLScheduler:
    def __init__(self, hm: HWModel, drv, fuse=True, vpu=None):
        self.hm = hm
        self.drv = drv
        self.fuse = fuse
        self.vpu = vpu          # 非线性运算的 RTL 执行者（l2/vpu_driver.py），None 表示在电脑上算
        self.host_ops = {}      # 留在电脑端执行的运算次数
        self._packed = {}

    # ---------------------------------------------------------------- 权重
    def _key(self, reqs):
        return tuple(id(r.lin) for r in reqs)

    def preload(self):
        """把（合并后的）全部权重打包好，相当于上板时预先加载到 DDR。"""
        t0 = time.time()
        for lins in self.hm.weight_groups():
            for group in self._groups([MatmulReq(lin, None, 0, "") for lin in lins]):
                self._get_packed(group)
        nbytes = sum(len(p.stream) for p in self._packed.values())
        return len(self._packed), nbytes, time.time() - t0

    def _groups(self, reqs):
        return [reqs] if self.fuse else [[r] for r in reqs]

    def _get_packed(self, group):
        key = self._key(group)
        p = self._packed.get(key)
        if p is None:
            wq = np.concatenate([r.lin.q for r in group])
            SW = np.concatenate([r.lin.SW for r in group])
            p = self._packed[key] = pk.Packed(wq, SW)
        return p

    # ---------------------------------------------------------------- 推理
    async def _run(self, reqs):
        out = []
        for group in self._groups(reqs):
            r0 = group[0]
            y = await self.drv.run(self._get_packed(group), r0.xq, r0.s)
            sizes = np.cumsum([r.lin.q.shape[0] for r in group])[:-1]
            out.extend(np.split(y, sizes))
        return out

    async def exec(self, op):
        """执行一个运算：矩阵乘交给 RTL；其余运算有 RTL 单元（self.vpu）就交给它，否则用 L0 在电脑上算。"""
        if op.kind == "matmul":
            return await self._run(op.args)
        if self.vpu is not None and op.kind in self.vpu.KINDS:
            return await self.vpu.exec(op)
        self.host_ops[op.kind] = self.host_ops.get(op.kind, 0) + 1
        return exec_op(op)

    async def forward_fx(self, token, pos):
        gen = self.hm.forward_steps(token, pos)
        res = None
        while True:
            try:
                op = gen.send(res)
            except StopIteration as e:
                return e.value
            res = await self.exec(op)

    async def generate(self, prompt_tokens, steps, on_token=None):
        """与 l0.run.generate 相同的贪心解码。返回 (tokens, 每步 logits)。"""
        self.hm.reset()
        tokens = [prompt_tokens[0]]
        all_logits = []
        for pos in range(steps):
            logits = await self.forward_fx(tokens[-1], pos)
            all_logits.append(logits)
            nxt = prompt_tokens[pos + 1] if pos + 1 < len(prompt_tokens) else int(np.argmax(logits))
            if nxt == Tokenizer.BOS:
                break
            tokens.append(nxt)
            if on_token:
                on_token(pos, nxt)
        return tokens, all_logits
