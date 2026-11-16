// MVM 计算核（可综合版）：y = sat32(round((W·x) * SW >> shift))，与 l0/fixedpoint.py 的 int_matmul 逐比特一致。
//
// 权重以 AXI-Stream 送入（每拍 = 一个 tile 的一行，N 个 int8，字节 j 在 tdata[8j+:8]）。
// 每个输出 tile（N 行）的流格式：
//   N/8 拍 SW（每拍 8 个 uint32，Q24）
//   KB*N 拍权重：kb 外层、行内层，即 W[tile*N + r, kb*N +: N]
// x 由外部同步读存储器提供：本模块在收拍当拍给出 x_idx，下一拍使用 x_blk。
//
// 结构：
//   乘加流水线 6 级（选数 → N 路乘法 → 4 路和 → 4 路和 → 2 路和 → 累加），从不停顿
//   两组累加器 / SW 交替使用：一组在输出时另一组接着累加，输出阶段不再挡住输入
//   输出流水线（读累加器 → ×SW 4 拍 → 加半个 LSB → 移位 → 饱和），随输出反压整体停顿
module mvm_core #(
    parameter int N   = 32,
    parameter int KBW = 6,    // kb_num 位宽，K 最多 (2^KBW-1)*N
    parameter int NTW = 16    // nt_num 位宽
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic [KBW-1:0]     kb_num,   // 输入维度块数（K/N），>=1
    input  logic [NTW-1:0]     nt_num,   // 输出 tile 数（M/N），>=1
    input  logic signed [7:0]  shift,    // = 24 - s，可为负（左移）
    output logic               busy,
    output logic               done,     // 任务完成单周期脉冲
    output logic [KBW-1:0]     x_idx,
    input  logic [N*8-1:0]     x_blk,    // x_idx 的下一拍有效
    input  logic [N*8-1:0]     w_tdata,
    input  logic               w_tvalid,
    output logic               w_tready,
    output logic [31:0]        y_tdata,
    output logic               y_tvalid,
    input  logic               y_tready,
    output logic               y_tlast
);
    localparam int SWB = N / 8;
    localparam int RW  = $clog2(N);
    localparam int SBW = $clog2(SWB + 1);

    // ------------------------------------------------ 状态
    typedef enum logic [1:0] {IIDLE, IWAIT, ISW, IW} ist_t;
    ist_t ist;
    logic           in_bank;
    logic [SBW-1:0] swb;
    logic [KBW-1:0] kb;
    logic [RW-1:0]  row;
    logic [NTW-1:0] in_tile;
    logic [1:0]     bbusy, bready;      // 每组：已被输入占用 / 累加完成待输出
    logic           job;
    logic [KBW-1:0] kbm1;            // 启动时预算 kb_num − 1、nt_num − 1
    logic [NTW-1:0] ntm1;

    logic [31:0]        sw  [2*N];
    logic signed [31:0] acc [2*N];

    logic               s1_v, s1_first, s1_tl, s1_bank;
    logic [RW-1:0]      s1_row;

    assign busy     = job;
    assign w_tready = (ist == ISW) || (ist == IW);
    assign x_idx    = kb;
    wire   fire     = w_tvalid && w_tready;

    // ------------------------------------------------ 乘加流水线（不停顿）
    // s1 选数 → s2 乘 → s3 四路和 → s4 四路和 → s5 两路和 → s6 累加
    localparam int L = 5;               // s1..s5 的附带信息
    logic [L:1]         pv, pfirst, ptl, pbank;
    logic [RW-1:0]      prow_ [L+1];
    logic [N*8-1:0]     s1_w;
    logic signed [15:0] s2_p [N];
    logic signed [17:0] s3_g [N/4];
    logic signed [19:0] s4_g [N/16];
    logic signed [20:0] s5_sum;

    // ------------------------------------------------ 输出流水线（随输出反压整体停顿）
    // q1 读累加器与 SW → 乘法（mul_pipe，4 拍）→ qa 加半个 LSB → qs 移位 → 饱和输出
    logic               p_act, pb, rel;
    logic [RW-1:0]      prow;
    logic [NTW-1:0]     ptile;
    logic               q1_v, q1_last, qm_v, qm_last, qa_v, qa_last, qs_v, qs_last;
    logic signed [31:0] q1_acc;
    logic [31:0]        q1_sw;
    logic signed [64:0] qm_p;
    logic signed [79:0] qa, qsh;
    logic signed [79:0] half_c;
    logic [4:0]         shr_c;
    logic [3:0]         shl_c;
    logic               neg_c;
    wire                adv = !y_tvalid || y_tready;

    logic signed [32:0] q1_swx;
    assign q1_swx = $signed({1'b0, q1_sw});
    mul_pipe #(.AW(32), .BW(33), .TW(1)) u_post (.clk, .en(adv), .vi(q1_v), .ti(q1_last),
        .a(q1_acc), .b(q1_swx), .vo(qm_v), .to(qm_last), .p(qm_p));

    assign rel = adv && p_act && (prow == RW'(N - 1));   // 本拍读完该组最后一行，释放

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (!rst_n) begin
            ist <= IIDLE; job <= 1'b0; bbusy <= '0; bready <= '0;
            s1_v <= 1'b0; pv <= '0;
            p_act <= 1'b0; q1_v <= 1'b0; qa_v <= 1'b0; qs_v <= 1'b0; y_tvalid <= 1'b0;
        end else begin
            // ---------------- 输入：SW 与权重
            s1_v <= 1'b0;
            case (ist)
                IIDLE: if (start) begin
                    ist <= IWAIT; job <= 1'b1;
                    in_tile <= '0; in_bank <= 1'b0; bbusy <= '0; bready <= '0;
                    pb <= 1'b0; ptile <= '0; p_act <= 1'b0;
                    neg_c <= (shift < 0);
                    kbm1 <= kb_num - 1'b1;
                    ntm1 <= nt_num - 1'b1;
                    shr_c <= (shift > 0) ? 5'(shift) : 5'd0;
                    shl_c <= (shift < 0) ? 4'(-shift) : 4'd0;
                    half_c <= (shift > 0) ? (80'sd1 <<< (shift - 8'sd1)) : 80'sd0;
                end
                IWAIT: if (!bbusy[in_bank]) begin ist <= ISW; swb <= '0; end
                ISW: if (fire) begin
                    for (int k = 0; k < 8; k++)
                        sw[{in_bank, RW'(int'(swb) * 8 + k)}] <= w_tdata[32*k +: 32];
                    if (swb == SBW'(SWB - 1)) begin ist <= IW; kb <= '0; row <= '0; end
                    else swb <= swb + 1'b1;
                end
                IW: if (fire) begin
                    s1_v <= 1'b1; s1_w <= w_tdata; s1_row <= row; s1_bank <= in_bank;
                    s1_first <= (kb == '0);
                    s1_tl <= (kb == kbm1) && (row == RW'(N - 1));
                    row <= row + 1'b1;
                    if (row == RW'(N - 1)) begin
                        if (kb == kbm1) begin
                            kb <= '0;
                            in_bank <= !in_bank;
                            in_tile <= in_tile + 1'b1;
                            ist <= (in_tile == ntm1) ? IIDLE : IWAIT;
                        end else
                            kb <= kb + 1'b1;
                    end
                end
                default: ist <= IIDLE;
            endcase

            // ---------------- 乘加流水线
            pv <= {pv[L-1:1], s1_v};
            pfirst <= {pfirst[L-1:1], s1_first}; ptl <= {ptl[L-1:1], s1_tl}; pbank <= {pbank[L-1:1], s1_bank};
            prow_[1] <= s1_row;
            for (int k = 2; k <= L; k++) prow_[k] <= prow_[k-1];
            for (int j = 0; j < N; j++)
                s2_p[j] <= 16'($signed(s1_w[8*j +: 8]) * $signed(x_blk[8*j +: 8]));
            for (int g = 0; g < N / 4; g++)
                s3_g[g] <= 18'(s2_p[4*g]) + 18'(s2_p[4*g+1]) + 18'(s2_p[4*g+2]) + 18'(s2_p[4*g+3]);
            for (int g = 0; g < N / 16; g++)
                s4_g[g] <= 20'(s3_g[4*g]) + 20'(s3_g[4*g+1]) + 20'(s3_g[4*g+2]) + 20'(s3_g[4*g+3]);
            s5_sum <= 21'(s4_g[0]) + 21'(s4_g[1]);
            if (pv[L-1])
                acc[{pbank[L-1], prow_[L-1]}] <= pfirst[L-1] ? 32'(s5_sum)
                                                             : acc[{pbank[L-1], prow_[L-1]}] + 32'(s5_sum);

            // ---------------- 组状态：输入占用 / 累加完成 / 输出释放
            if (ist == IWAIT && !bbusy[in_bank]) bbusy[in_bank] <= 1'b1;
            if (pv[L-1] && ptl[L-1]) bready[pbank[L-1]] <= 1'b1;
            if (rel) begin bbusy[pb] <= 1'b0; bready[pb] <= 1'b0; end

            // ---------------- 输出流水线
            if (adv) begin
                qa_v <= qm_v; qa_last <= qm_last; qa <= 80'(qm_p) + half_c;
                qs_v <= qa_v; qs_last <= qa_last; qsh <= neg_c ? (qa <<< shl_c) : (qa >>> shr_c);
                y_tvalid <= qs_v; y_tlast <= qs_last;
                // 饱和：高位 [79:31] 不全相同即溢出
                if (!(&qsh[79:31]) && (|qsh[79:31])) y_tdata <= qsh[79] ? 32'h8000_0000 : 32'h7fff_ffff;
                else y_tdata <= qsh[31:0];
                if (y_tvalid && y_tready && y_tlast) begin done <= 1'b1; job <= 1'b0; end
                if (p_act) begin
                    q1_v <= 1'b1;
                    q1_acc <= acc[{pb, prow}];
                    q1_sw <= sw[{pb, prow}];
                    q1_last <= (prow == RW'(N - 1)) && (ptile == ntm1);
                    prow <= prow + 1'b1;
                    if (prow == RW'(N - 1)) begin
                        p_act <= 1'b0; pb <= !pb; ptile <= ptile + 1'b1;
                    end
                end else
                    q1_v <= 1'b0;
            end
            if (!p_act && bready[pb] && job) begin p_act <= 1'b1; prow <= '0; end
        end
    end
endmodule
