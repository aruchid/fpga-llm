"""从 l0/fixedpoint.py 的查找表生成 rtl/vpu_common.svh，保证 RTL 与 L0 用同一份表。

查找表写成 case 函数：Verilator、yosys、Vivado 都能直接综合成 ROM（放在寄存器级之前即推断为 BRAM/LUTRAM）。
该文件在每个向量单元的 module 内部 `include。

  python -m l2.gen_luts
"""
from pathlib import Path

from l0 import fixedpoint as fp

OUT = Path(__file__).resolve().parents[1] / "rtl" / "vpu_common.svh"


def rom(name, vals, aw, dw):
    vals = [int(v) for v in vals]
    assert all(0 <= v < (1 << dw) for v in vals), name
    lines = [f"function automatic logic [{dw - 1}:0] {name}(input logic [{aw - 1}:0] i);",
             "    case (i)"]
    for i, v in enumerate(vals):
        lines.append(f"        {aw}'d{i}: {name} = {dw}'d{v};")
    lines += [f"        default: {name} = '0;", "    endcase", "endfunction", ""]
    return "\n".join(lines)


HELPERS = """
// x·2^-s 并舍入（加半个 LSB 后算术右移）；s<0 为左移。调用方保证结果不溢出 64 位
function automatic logic signed [63:0] rsr64(input logic signed [63:0] xv, input int s);
    if (s > 0)
        rsr64 = (xv + (64'sd1 <<< (s - 1))) >>> s;
    else if (s == 0)
        rsr64 = xv;
    else
        rsr64 = xv <<< (-s);
endfunction

// 非负数的位宽（Python 的 int.bit_length），优先编码器
function automatic int bitlen64(input logic [63:0] mv);
    bitlen64 = 0;
    for (int i = 0; i < 64; i++)
        if (mv[i]) bitlen64 = i + 1;
endfunction

// 两级求位宽：hibyte64 找最高非零字节（{有效, 序号}），bitlen8 求字节内位宽
function automatic logic [3:0] hibyte64(input logic [63:0] mv);
    hibyte64 = 4'd0;
    for (int k = 0; k < 8; k++)
        if (|mv[8*k +: 8]) hibyte64 = {1'b1, 3'(k)};
endfunction

function automatic logic [3:0] bitlen8(input logic [7:0] bv);
    bitlen8 = 4'd0;
    for (int i = 0; i < 8; i++)
        if (bv[i]) bitlen8 = 4'(i + 1);
endfunction

function automatic logic signed [31:0] sat32(input logic signed [79:0] xv);
    if (xv > 80'sd2147483647) sat32 = 32'sh7fff_ffff;
    else if (xv < -80'sd2147483648) sat32 = 32'sh8000_0000;
    else sat32 = xv[31:0];
endfunction
"""


def main():
    text = ("// 自动生成：python -m l2.gen_luts（查找表来源 l0/fixedpoint.py），不要手改。\n"
            "// 在 module 内部 `include 使用。\n"
            "/* verilator lint_off UNUSEDPARAM */\n"
            f"localparam int LOG2E_Q16 = {fp.LOG2E_Q16};\n"
            f"localparam int EPS_Q16 = {fp.EPS_Q16};\n"
            "/* verilator lint_on UNUSEDPARAM */\n"
            "/* verilator lint_off UNUSEDSIGNAL */\n"
            + rom("rsqrt_lut", fp.RSQRT_LUT, 8, 17)
            + rom("recip_lut", fp.RECIP_LUT, 8, 17)
            + rom("exp2_lut", fp.EXP2_LUT, 9, 18)
            + rom("sig_lut", fp.SIG_LUT, 10, 17)
            + HELPERS
            + "/* verilator lint_on UNUSEDSIGNAL */\n")
    OUT.write_text(text)
    print(f"写入 {OUT}")


if __name__ == "__main__":
    main()
