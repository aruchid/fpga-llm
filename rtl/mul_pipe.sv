// 有符号乘法流水线：A×B 拆成若干 ≤25×18 的部分积（各对应 1 个 DSP48），再逐级两两相加。
// 延迟 = 2 + ceil(log2(部分积个数)) 拍；en=0 时整体停顿（接可反压的流水线）。
// vi/ti 与数据同步进入，vo/to 与结果同步输出，便于携带有效位和附带信息。
/* verilator lint_off DECLFILENAME */
module mul_pipe #(
    parameter int AW = 32,
    parameter int BW = 32,
    parameter int TW = 1
) (
    input  logic                    clk,
    input  logic                    en,
    input  logic                    vi,
    input  logic [TW-1:0]           ti,
    input  logic signed [AW-1:0]    a,
    input  logic signed [BW-1:0]    b,
    output logic                    vo,
    output logic [TW-1:0]           to,
    output logic signed [AW+BW-1:0] p
);
    localparam int NA  = (AW + 22) / 24;          // A：低位每块 24 位（无符号），最高块带符号 ≤ 25 位
    localparam int NB  = (BW + 15) / 17;          // B：低位每块 17 位（无符号），最高块带符号 ≤ 18 位
    localparam int NP  = NA * NB;
    localparam int LV  = (NP <= 1) ? 0 : $clog2(NP);
    localparam int NPP = 1 << LV;
    localparam int PW  = AW + BW;

    logic signed [AW-1:0] ar;
    logic signed [BW-1:0] br;
    logic signed [24:0]   ca [NA];
    logic signed [17:0]   cb [NB];
    logic signed [PW-1:0] tr [(LV+1)*NPP];
    logic [LV+1:0]        vs;
    logic [TW-1:0]        ts [LV+2];

    genvar gi, gj;
    generate
        for (gi = 0; gi < NA; gi++) begin : g_a
            if (gi < NA - 1) begin : g_lo
                assign ca[gi] = $signed({1'b0, ar[24*gi +: 24]});
            end else begin : g_hi
                assign ca[gi] = 25'($signed(ar[AW-1:24*gi]));
            end
        end
        for (gj = 0; gj < NB; gj++) begin : g_b
            if (gj < NB - 1) begin : g_lo
                assign cb[gj] = $signed({1'b0, br[17*gj +: 17]});
            end else begin : g_hi
                assign cb[gj] = 18'($signed(br[BW-1:17*gj]));
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (en) begin
            ar <= a;
            br <= b;
            vs <= {vs[LV:0], vi};
            ts[0] <= ti;
            for (int k = 1; k < LV + 2; k++) ts[k] <= ts[k-1];
            for (int i = 0; i < NA; i++)
                for (int j = 0; j < NB; j++)
                    tr[i*NB + j] <= (PW'(ca[i]) * PW'(cb[j])) <<< (24*i + 17*j);
            for (int k = NP; k < NPP; k++) tr[k] <= '0;
            for (int l = 0; l < LV; l++)
                for (int k = 0; k < (NPP >> (l + 1)); k++)
                    tr[(l+1)*NPP + k] <= tr[l*NPP + 2*k] + tr[l*NPP + 2*k + 1];
        end
    end
    assign p  = tr[LV*NPP];
    assign vo = vs[LV+1];
    assign to = ts[LV+1];
endmodule
