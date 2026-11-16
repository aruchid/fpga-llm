"""粗略时序估算：读 yosys synth_xilinx 后的 JSON 网表，按典型 7 系列（-2 速度等级）延迟找最长的寄存器到寄存器路径。

只用来找出需要插流水线的地方、估计量级；不能代替 Vivado 的布局布线后时序。

延迟模型（含布线，单位 ns）：
  LUT1~6 0.9；CARRY4 链内 0.15、进链 0.6；MUXF7/F8 0.3；LUTRAM 异步读 1.0
  DSP48E1：内部无寄存器时组合穿过 3.2；有寄存器时视为路径起止点
  起点：FD* clk→Q 0.5，BRAM clk→DO 2.2，DSP 寄存器输出 0.6
  终点：建立时间 FD* 0.1，BRAM 0.6，DSP 0.5

  python synth/sta_est.py synth/mvm_top_xc7.json
"""
import json
import sys
from collections import defaultdict

LUT = {f"LUT{i}" for i in range(1, 7)}
FF = {"FDRE", "FDSE", "FDCE", "FDPE"}
BRAM = {"RAMB18E1", "RAMB36E1"}
LUTRAM = {"RAM32M", "RAM64M", "RAM32X1D", "RAM64X1D", "RAM128X1D", "RAM32M16", "RAM64M8"}


def dsp_registered(params):
    return any(int(str(params.get(p, "0")), 2 if set(str(params.get(p, "0"))) <= {"0", "1"} else 10) != 0
               for p in ("MREG", "PREG"))


def analyze(path):
    d = json.load(open(path))
    top = next(m for m in d["modules"].values() if m.get("attributes", {}).get("top"))
    cells = top["cells"]
    names = {}          # 优先用源码里的信号名（不以 $ 开头）
    for nm, n in top["netnames"].items():
        for i, b in enumerate(n["bits"]):
            if not isinstance(b, int):
                continue
            label = f"{nm}[{i}]" if len(n["bits"]) > 1 else nm
            if b not in names or (names[b].startswith("$") and not nm.startswith("$")):
                names[b] = label

    driver = {}        # bit → (cell, delay through cell)
    sinks = defaultdict(list)
    start = {}         # bit → arrival time (起点)
    end_setup = []     # (bit, setup)
    comb = {}          # cell → (inputs, outputs, delay)
    srcs = {}
    for cn, c in cells.items():
        t = c["type"]
        dirs = c.get("port_directions", {})
        ins = [b for p, bs in c["connections"].items() if dirs.get(p) == "input" for b in bs if isinstance(b, int)]
        outs = [b for p, bs in c["connections"].items() if dirs.get(p) == "output" for b in bs if isinstance(b, int)]
        if t in FF:
            for b in outs: start[b] = 0.5
            data = [b for p, bs in c["connections"].items() if p in ("D", "R", "S", "CE") for b in bs if isinstance(b, int)]
            end_setup += [(b, 0.1, cn) for b in data]
        elif t in BRAM:
            for b in outs: start[b] = 2.2
            end_setup += [(b, 0.6, cn) for b in ins]
        elif t == "DSP48E1" and dsp_registered(c.get("parameters", {})):
            for b in outs: start[b] = 0.6
            end_setup += [(b, 0.5, cn) for b in ins]
        else:
            if t in LUT: dl = 0.9
            elif t == "CARRY4": dl = 0.15
            elif t in ("MUXF7", "MUXF8"): dl = 0.3
            elif t in LUTRAM: dl = 1.0
            elif t == "DSP48E1": dl = 3.2
            else: dl = 0.0
            comb[cn] = (ins, outs, dl, t)
        srcs[cn] = c.get("attributes", {}).get("src", "")
    for cn, (ins, outs, dl, t) in comb.items():
        for b in ins: sinks[b].append(cn)
        for b in outs: driver[b] = cn

    # 拓扑序 DP
    arr = dict(start)
    for p, info in top["ports"].items():
        if info["direction"] == "input":
            for b in info["bits"]:
                if isinstance(b, int): arr[b] = 0.0
    prev = {}
    indeg = {cn: sum(1 for b in ins if b in driver) for cn, (ins, _, _, _) in comb.items()}
    ready = [cn for cn, k in indeg.items() if k == 0]
    while ready:
        cn = ready.pop()
        ins, outs, dl, t = comb[cn]
        best, bsrc = 0.0, None
        for b in ins:
            a = arr.get(b, 0.0)
            # 进位链入口额外加布线
            if t == "CARRY4" and comb.get(driver.get(b, ""), (0, 0, 0, ""))[3] != "CARRY4":
                a += 0.45
            if a > best: best, bsrc = a, b
        for b in outs:
            arr[b] = best + dl
            prev[b] = (cn, bsrc)
            for s in sinks[b]:
                indeg[s] -= 1
                if indeg[s] == 0: ready.append(s)
    worst = max(((arr.get(b, 0.0) + su, b, cn) for b, su, cn in end_setup), default=(0, None, None))
    # 回溯路径上的信号名
    trail, b = [], worst[1]
    while b is not None and b in prev:
        cn, b2 = prev[b]
        trail.append((names.get(b, "?"), comb[cn][3], srcs.get(cn, "")))
        b = b2
    if b is not None: trail.append((names.get(b, "?"), "起点", ""))
    return worst[0], worst[2], trail[::-1]


if __name__ == "__main__":
    for f in sys.argv[1:]:
        t, endcell, trail = analyze(f)
        print(f"{f}: 最长寄存器间路径 ≈ {t:.2f} ns（≈ {1000 / t:.0f} MHz）")
        kinds = defaultdict(int)
        for _, k, _ in trail: kinds[k] += 1
        print("   经过：", dict(kinds))
        lines = []
        for _, _, src in trail:
            for part in src.split("|"):
                part = part.split("/")[-1].split(".")[0:2]
                ln = ".".join(part)
                if ln and ln not in lines: lines.append(ln)
        print("   源码：", " → ".join(lines[:12]))
        pub = [n for n, _, _ in trail if not n.startswith("$")]
        print("   信号：", " → ".join(dict.fromkeys(pub)) if pub else "（无源码信号名）")
