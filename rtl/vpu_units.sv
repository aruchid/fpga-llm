// 向量单元（可综合、按时序切分流水线）：模型里除权重矩阵乘以外的全部运算，与 l0/fixedpoint.py 逐比特一致。
//
// 接口：AXI-Stream，元素 64 位有符号（小端），向量以 tlast 结尾。
// 实现：
//   - 向量缓冲为同步读写的块 RAM
//   - 数据通路为定长流水线 + 输出 FIFO：按"在途 + FIFO 占用 < FIFO 深度"发射，支持任意反压
//   - 乘法用 mul_pipe（拆成 ≤25×18 的部分积逐级相加）；舍入、变量移位、饱和各占一级
//   - 1/√x、1/x 为时序电路，各用一个乘法流水线
// 位宽按模型实际数值范围确定：
//   - quant 输入 |x| < 2^39（RoPE 后最大约 2^33）
//   - rmsnorm 的 x、g，silu_gate / resadd / rope 的输入为 int32（来自饱和后的矩阵乘或残差）
//   - softmax / 注意力分数 < 2^62，注意力的移位量 qs + ks ≤ 56
//
//   vpu_quant    quantize_shift：Q16 向量 → int8 + 移位量 s
//   vpu_rmsnorm  rmsnorm_fx：x（a 流）、增益 g（b 流）
//   vpu_softmax  softmax_fx
//   vpu_ew       逐元素：op=0 残差相加 sat32(a+b)，op=1 SwiGLU 门控 silu(a)·b
//   vpu_rope     rope_fx：a 流为成对的 v0,v1；b 流每对一个 {sin[31:0], cos[31:0]}（Q14）
//   attn_head    一个注意力头：q·K → 缩放 → Softmax → P·V，16 路并行，每行 4 拍
/* verilator lint_off DECLFILENAME */
// 各处中间结果按最宽情况声明、只取用到的低位，属有意为之
/* verilator lint_off UNUSEDSIGNAL */

// ============================================================ 同步 FIFO（首字直出）
module vfifo #(parameter int W = 64, parameter int D = 8) (
    input  logic                 clk, rst_n,
    input  logic                 push,
    input  logic [W-1:0]         din,
    input  logic                 pop,
    output logic [W-1:0]         dout,
    output logic                 empty,
    output logic [$clog2(D):0]   cnt
);
    localparam int PW = $clog2(D);
    logic [W-1:0]  m [D];
    logic [PW-1:0] wp, rp;
    assign dout  = m[rp];
    assign empty = (cnt == 0);
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wp <= '0; rp <= '0; cnt <= '0;
        end else begin
            if (push) begin m[wp] <= din; wp <= wp + 1'b1; end
            if (pop) rp <= rp + 1'b1;
            cnt <= cnt + (push ? (PW+1)'(1) : (PW+1)'(0)) - (pop ? (PW+1)'(1) : (PW+1)'(0));
        end
    end
endmodule

// ============================================================ 发射计数：在途个数
module vcredit #(parameter int D = 16) (
    input  logic                 clk, rst_n,
    input  logic                 inc, dec,
    input  logic [$clog2(D):0]   f_cnt,
    output logic                 ok          // 还能再发射一个
);
    localparam int CW = $clog2(D) + 1;
    logic [CW-1:0] pend;
    // ok 晚一拍：上一拍的"在途 + 占用"再加本拍可能的 1 个发射仍不超过 D
    always_ff @(posedge clk) begin
        if (!rst_n) begin pend <= '0; ok <= 1'b0; end
        else begin
            pend <= pend + (inc ? CW'(1) : CW'(0)) - (dec ? CW'(1) : CW'(0));
            ok <= (32'(pend) + 32'(f_cnt) + (inc ? 32'd1 : 32'd0) < D - 2);
        end
    end
endmodule

// ============================================================ 1/√m（Q16 → Q16），时序实现
module vpu_rsqrt (
    input  logic        clk, rst_n,
    input  logic        start,
    input  logic [63:0] m,
    output logic        done,
    output logic [31:0] y
);
    `include "vpu_common.svh"
    typedef enum logic [4:0] {IDLE, NZ, BH, BL, EV, SH, N1, NORM, LUT, M1, W1, M2, W2, M3, W3, OA, OB} st_t;
    st_t st;
    logic [3:0]         hb;
    logic [7:0]         hbyte;
    logic               neg;
    logic [5:0]         sha;
    logic [63:0]        m_r, m_c;
    logic [7:0]         nz;
    logic [3:0]         hb_c;
    always_comb begin
        hb_c = 4'd0;
        for (int k = 0; k < 8; k++) if (nz[k]) hb_c = {1'b1, 3'(k)};
    end
    logic [31:0]        n_r;
    logic signed [7:0]  e_r, k_r;
    logic [6:0]         bl_r;
    logic signed [19:0] yv, yy;
    logic signed [23:0] t;
    logic [17:0]        u16;
    logic               it, mv, mvo;
    logic signed [23:0] ma, mb;
    logic signed [47:0] mp;
    logic signed [31:0] ya;
    logic               kneg;
    logic [4:0]         kamt;

    mul_pipe #(.AW(24), .BW(24)) u_m (.clk, .en(1'b1), .vi(mv), .ti(1'b0), .a(ma), .b(mb),
                                      .vo(mvo), .to(), .p(mp));

    always_ff @(posedge clk) begin
        done <= 1'b0;
        mv <= 1'b0;
        if (!rst_n) st <= IDLE;
        else case (st)
            IDLE: if (start) begin m_r <= m; st <= NZ; end
            NZ:   begin for (int k = 0; k < 8; k++) nz[k] <= |m_r[8*k +: 8]; st <= BH; end
            BH:   begin hb <= hb_c; hbyte <= m_r[8*hb_c[2:0] +: 8]; st <= BL; end
            BL:   begin bl_r <= hb[3] ? 7'({hb[2:0], 3'd0}) + 7'(bitlen8(hbyte)) : 7'd0; st <= EV; end
            EV:   begin                                     // e = bl − 31，取偶数（向 −∞）
                e_r <= (8'($signed({1'b0, bl_r})) - 8'sd31) & ~8'sd1;
                st <= SH;
            end
            SH:   begin neg <= (e_r < 0); sha <= (e_r < 0) ? 6'(-e_r) : 6'(e_r); k_r <= (e_r + 8'sd14) >>> 1; st <= N1; end
            N1:   begin m_c <= neg ? (m_r << {sha[5:3], 3'd0}) : (m_r >> {sha[5:3], 3'd0}); st <= NORM; end
            NORM: begin                                     // n ∈ [2^30, 2^32)；k = (e + 30 − 16) / 2
                n_r <= neg ? 32'(m_c << sha[2:0]) : 32'(m_c >> sha[2:0]);
                st <= LUT;
            end
            LUT:  begin
                yv <= 20'($signed({1'b0, rsqrt_lut(n_r[31:24] - 8'd64)}));
                u16 <= n_r[31:14];
                it <= 1'b0;
                st <= M1;
            end
            M1:   begin ma <= 24'(yv); mb <= 24'(yv); mv <= 1'b1; st <= W1; end
            W1:   if (mvo) begin yy <= 20'(mp >>> 16); st <= M2; end
            M2:   begin ma <= $signed({6'd0, u16}); mb <= 24'(yy); mv <= 1'b1; st <= W2; end
            W2:   if (mvo) begin t <= 24'(mp >>> 16); st <= M3; end
            M3:   begin ma <= 24'(yv); mb <= 24'sd196608 - t; mv <= 1'b1; st <= W3; end   // y·(3 − u·y²)/2
            W3:   if (mvo) begin
                yv <= 20'(mp >>> 17);
                it <= 1'b1;
                st <= it ? OA : M1;
            end
            OA:   begin
                ya <= (k_r > 0) ? 32'(yv) + (32'sd1 <<< (k_r - 8'sd1)) : 32'(yv);
                kneg <= (k_r < 0); kamt <= (k_r < 0) ? 5'(-k_r) : 5'(k_r);
                st <= OB;
            end
            OB:   begin y <= kneg ? 32'(ya <<< kamt) : 32'(ya >>> kamt); done <= 1'b1; st <= IDLE; end
            default: st <= IDLE;
        endcase
    end
endmodule

// ============================================================ 1/m（Q16 → Q32），时序实现
module vpu_recip (
    input  logic        clk, rst_n,
    input  logic        start,
    input  logic [39:0] m,                // ≥ 2^16（Softmax 的和至少为 1.0）
    output logic        done,
    output logic [35:0] y
);
    `include "vpu_common.svh"
    typedef enum logic [3:0] {IDLE, NZ, BH, BL, SH, N1, NORM, LUT, M1, W1, M2, W2, OA, OB} st_t;
    st_t st;
    logic [3:0]         hb;
    logic [7:0]         hbyte;
    logic               neg;
    logic [5:0]         sha;
    logic [39:0]        m_r, m_c;
    logic [4:0]         nz;
    logic [3:0]         hb_c;
    always_comb begin
        hb_c = 4'd0;
        for (int k = 0; k < 5; k++) if (nz[k]) hb_c = {1'b1, 3'(k)};
    end
    logic [30:0]        n_r;
    logic signed [7:0]  e_r, k_r;
    logic signed [19:0] yv;
    logic signed [23:0] t;
    logic [16:0]        u16;
    logic               it, mv, mvo;
    logic signed [23:0] ma, mb;
    logic signed [47:0] mp;
    logic signed [39:0] ya;
    logic               kneg;
    logic [4:0]         kamt;

    mul_pipe #(.AW(24), .BW(24)) u_m (.clk, .en(1'b1), .vi(mv), .ti(1'b0), .a(ma), .b(mb),
                                      .vo(mvo), .to(), .p(mp));

    always_ff @(posedge clk) begin
        done <= 1'b0;
        mv <= 1'b0;
        if (!rst_n) st <= IDLE;
        else case (st)
            IDLE: if (start) begin m_r <= m; st <= NZ; end
            NZ:   begin for (int k = 0; k < 5; k++) nz[k] <= |m_r[8*k +: 8]; st <= BH; end
            BH:   begin hb <= hb_c; hbyte <= m_r[8*hb_c[2:0] +: 8]; st <= BL; end
            BL:   begin e_r <= (hb[3] ? 8'({hb[2:0], 3'd0}) + 8'(bitlen8(hbyte)) : 8'd0) - 8'sd31; st <= SH; end
            SH:   begin neg <= (e_r < 0); sha <= (e_r < 0) ? 6'(-e_r) : 6'(e_r); k_r <= e_r - 8'sd2; st <= N1; end
            N1:   begin m_c <= neg ? (m_r << {sha[5:3], 3'd0}) : (m_r >> {sha[5:3], 3'd0}); st <= NORM; end
            NORM: begin                                     // n ∈ [2^30, 2^31)；k = e + 30 − 16 − 16
                n_r <= neg ? 31'(m_c << sha[2:0]) : 31'(m_c >> sha[2:0]);
                st <= LUT;
            end
            LUT:  begin
                yv <= 20'($signed({1'b0, recip_lut(n_r[29:22])}));
                u16 <= n_r[30:14];
                it <= 1'b0;
                st <= M1;
            end
            M1:   begin ma <= $signed({7'd0, u16}); mb <= 24'(yv); mv <= 1'b1; st <= W1; end
            W1:   if (mvo) begin t <= 24'(mp >>> 16); st <= M2; end
            M2:   begin ma <= 24'(yv); mb <= 24'sd131072 - t; mv <= 1'b1; st <= W2; end   // y·(2 − u·y)
            W2:   if (mvo) begin
                yv <= 20'(mp >>> 16);
                it <= 1'b1;
                st <= it ? OA : M1;
            end
            OA:   begin
                ya <= (k_r > 0) ? 40'(yv) + (40'sd1 <<< (k_r - 8'sd1)) : 40'(yv);
                kneg <= (k_r < 0); kamt <= (k_r < 0) ? 5'(-k_r) : 5'(k_r);
                st <= OB;
            end
            OB:   begin y <= kneg ? 36'(ya <<< kamt) : 36'(ya >>> kamt); done <= 1'b1; st <= IDLE; end
            default: st <= IDLE;
        endcase
    end
endmodule

// ============================================================ quantize_shift
module vpu_quant #(parameter int DEPTH = 2048) (
    input  logic        clk, rst_n,
    input  logic [63:0] a_tdata,
    input  logic        a_tvalid, a_tlast,
    output logic        a_tready,
    output logic [63:0] y_tdata,
    output logic        y_tvalid, y_tlast,
    input  logic        y_tready,
    output logic [7:0]  s_out           // 本次向量的移位量（输出期间有效，保持到下一次）
);
    `include "vpu_common.svh"
    localparam int AW = $clog2(DEPTH);
    localparam int FD = 16;
    typedef enum logic [3:0] {LOAD, DRAIN, CZ, C0, C1, C2, C3, C4, C5, C6, OUTP, OUT} st_t;
    st_t st;

    logic signed [39:0] mem [DEPTH];
    logic signed [39:0] rd;
    logic [AW-1:0]      ra;
    logic [AW:0]        cnt, n, iss;
    logic               l1v, l1l, l2v;
    logic signed [39:0] l1x;
    logic [40:0]        l2a, m;
    logic [6:0]         bl;
    logic [3:0]         hb, hb_c;
    logic [7:0]         hbyte;
    logic [5:0]         nz;
    logic [41:0]        h0, mhs;
    logic [7:0]         s0, s;
    logic [41:0]        mh;
    logic [40:0]        half;
    logic               ok, issue, pop, f_empty;
    logic               v1, v2, v3, v4, t1, t2, t3, t4;
    logic signed [40:0] xa;
    logic signed [40:0] xs;
    logic [7:0]         q4;
    logic [$clog2(FD):0] f_cnt;
    logic [8:0]         f_dout;

    assign a_tready = (st == LOAD);
    assign issue = (st == OUT) && (iss < n) && ok;
    assign ra = iss[AW-1:0];
    assign y_tvalid = (st == OUT) && !f_empty;
    assign pop = y_tvalid && y_tready;
    assign y_tdata = 64'($signed(f_dout[7:0]));
    assign y_tlast = f_dout[8];
    assign s = ((mh >> s0) > 127) ? s0 + 8'd1 : s0;
    always_comb begin
        hb_c = 4'd0;
        for (int k = 0; k < 6; k++) if (nz[k]) hb_c = {1'b1, 3'(k)};
    end       // rsr(m, s0) > 127 时 s 再加 1

    always_ff @(posedge clk) begin
        if (l1v) mem[cnt[AW-1:0]] <= l1x;
        rd <= mem[ra];
    end

    vcredit #(.D(FD)) u_c (.clk, .rst_n, .inc(issue), .dec(v4), .f_cnt, .ok);
    vfifo #(.W(9), .D(FD)) u_f (.clk, .rst_n, .push(v4), .din({t4, q4}), .pop,
                                 .dout(f_dout), .empty(f_empty), .cnt(f_cnt));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            st <= LOAD; cnt <= '0; n <= '0; m <= '0; s_out <= '0; iss <= '0;
            l1v <= 1'b0; l2v <= 1'b0; v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
        end else begin
            // 载入流水线：寄存 → 取绝对值 → 比较最大值
            l1v <= (st == LOAD) && a_tvalid; l1x <= a_tdata[39:0]; l1l <= a_tlast;
            l2v <= l1v; l2a <= l1x[39] ? 41'(-l1x) : 41'(l1x);
            if (l2v && l2a > m) m <= l2a;
            if (l1v) cnt <= cnt + 1'b1;
            // 输出流水线：读 → 加半个 LSB → 移位 → 截到 ±127
            v1 <= issue; t1 <= (iss == n - 1'b1);
            v2 <= v1; t2 <= t1; xa <= 41'(rd) + 41'(half);
            v3 <= v2; t3 <= t2; xs <= xa >>> s_out;
            v4 <= v3; t4 <= t3; q4 <= (xs > 127) ? 8'sd127 : (xs < -127) ? -8'sd127 : 8'(xs);
            case (st)
                LOAD: if (l1v && l1l) begin n <= cnt + 1'b1; st <= DRAIN; end
                DRAIN: st <= CZ;                                // 等最后一个绝对值比较完
                CZ: begin for (int k = 0; k < 6; k++) nz[k] <= |m[8*k +: 8]; st <= C0; end
                C0: begin hb <= hb_c; hbyte <= m[8*hb_c[2:0] +: 8]; st <= C1; end
                C1: begin bl <= hb[3] ? 7'({hb[2:0], 3'd0}) + 7'(bitlen8(hbyte)) : 7'd0; st <= C2; end
                C2: begin s0 <= (bl > 7) ? 8'(bl - 7'd7) : 8'd0; st <= C3; end
                C3: begin h0 <= (s0 > 0) ? (42'd1 << (s0 - 8'd1)) : 42'd0; st <= C4; end
                C4: begin mh <= 42'(m) + h0; st <= C5; end
                C5: begin mhs <= mh >> s0; st <= C6; end
                C6: begin s_out <= s; st <= OUTP; end
                OUTP: begin
                    half <= (s_out > 0) ? (41'd1 << (s_out - 8'd1)) : 41'd0;
                    iss <= '0;
                    st <= OUT;
                end
                OUT: begin
                    if (issue) iss <= iss + 1'b1;
                    if (pop && f_dout[8]) begin st <= LOAD; cnt <= '0; m <= '0; end
                end
                default: st <= LOAD;
            endcase
        end
    end
endmodule

// ============================================================ 指数流水线（Softmax 与注意力共用）
// e = 2^((x − mx)·log2e)（Q16）。x − mx < −2^20 时结果必为 0，先截到 −2^20 不改变结果，
// 但让后续乘法只需 21 位。延迟 11 拍，不停顿。
module vpu_exp_pipe (
    input  logic               clk,
    input  logic               vi,
    input  logic signed [63:0] x,
    input  logic signed [63:0] mx,
    output logic               vo,
    output logic [17:0]        e
);
    `include "vpu_common.svh"
    logic [10:0]        v;
    logic [17:0]        df7;
    logic [7:0]         fr8;
    logic [4:0]         sh10;
    logic [17:0]        a8;
    logic signed [64:0] d;
    logic signed [20:0] dc;
    logic signed [37:0] mlt;
    logic signed [21:0] t;
    logic signed [21:0] ip;
    logic [15:0]        f;
    logic [8:0]         idx;
    logic [7:0]         fr6, fr7;
    logic [4:0]         sh5, sh6, sh7, sh8, sh9;
    logic [17:0]        a6, b6, a7;
    logic [25:0]        dm7;
    logic [17:0]        v8;
    logic [41:0]        vh9;
    assign ip = t >>> 16;                       // ∈ [−24, 0]
    assign f  = 16'(t - (ip <<< 16));
    always_ff @(posedge clk) begin
        v   <= {v[9:0], vi};
        d   <= 65'(x) - 65'(mx);                                            // 1
        dc  <= (d < -65'sd1048576) ? -21'sd1048576 : 21'(d);                // 2
        mlt <= 38'(dc) * 38'(LOG2E_Q16);                                     // 3
        t   <= 22'((mlt + 38'sd32768) >>> 16);                               // 4  t = rsr(d·log2e, 16)
        idx <= 9'(f[15:8]); fr6 <= f[7:0]; sh5 <= 5'(-ip);                   // 5
        a6  <= exp2_lut(idx); b6 <= exp2_lut(idx + 9'd1); sh6 <= sh5;        // 6  查表
        fr7 <= fr6;
        df7 <= b6 - a6; a7 <= a6; sh7 <= sh6; fr8 <= fr7;                    // 7  表差
        dm7 <= 26'(df7) * 26'(fr8); a8 <= a7; sh8 <= sh7;                    // 8  插值乘
        v8  <= a8 + 18'(dm7 >> 8); sh10 <= sh8;                              // 9  插值加
        vh9 <= 42'(v8) + ((sh10 > 0) ? (42'd1 << (sh10 - 5'd1)) : 42'd0);    // 10 rsr(v, −ip)
        sh9 <= sh10;
        e   <= 18'(vh9 >> sh9);                                              // 11
    end
    assign vo = v[10];
endmodule

// ============================================================ Softmax
module vpu_softmax #(parameter int DEPTH = 2048) (
    input  logic        clk, rst_n,
    input  logic [63:0] a_tdata,
    input  logic        a_tvalid, a_tlast,
    output logic        a_tready,
    output logic [63:0] y_tdata,
    output logic        y_tvalid, y_tlast,
    input  logic        y_tready
);
    `include "vpu_common.svh"
    localparam int AW = $clog2(DEPTH);
    localparam int FD = 16;
    typedef enum logic [2:0] {LOAD, EXP, SFU, OUT} st_t;
    st_t st;

    logic signed [63:0] mem [DEPTH];
    logic signed [63:0] rd, mx;
    logic [AW-1:0]      ra, wa;
    logic               we;
    logic [63:0]        wd;
    logic [AW:0]        cnt, n, iss, wcnt;
    logic               l1v, l1l;
    logic signed [63:0] l1x;
    logic               ei, eo;
    logic [17:0]        e;
    logic [39:0]        total;
    logic               rc_start, rc_done;
    logic [35:0]        rc_y, r;
    logic               ok, issue, pop, f_empty, v1, t1, mvo, mto, v6, t6;
    logic signed [55:0] mp;
    logic [17:0]        p6;
    logic [$clog2(FD):0] f_cnt;
    logic [18:0]        f_dout;

    assign a_tready = (st == LOAD);
    assign issue = ((st == EXP) && (iss < n)) || ((st == OUT) && (iss < n) && ok);
    assign ra = iss[AW-1:0];
    assign y_tvalid = (st == OUT) && !f_empty;
    assign pop = y_tvalid && y_tready;
    assign y_tdata = 64'(f_dout[17:0]);
    assign y_tlast = f_dout[18];

    always_comb begin
        we = 1'b0; wa = cnt[AW-1:0]; wd = l1x;
        if (l1v) we = 1'b1;
        else if (eo) begin we = 1'b1; wa = wcnt[AW-1:0]; wd = 64'(e); end
    end
    always_ff @(posedge clk) begin
        if (we) mem[wa] <= wd;
        rd <= mem[ra];
    end

    vpu_exp_pipe u_exp (.clk, .vi(ei), .x(rd), .mx, .vo(eo), .e);
    vpu_recip u_recip (.clk, .rst_n, .start(rc_start), .m(total), .done(rc_done), .y(rc_y));
    logic signed [18:0] sm_a;
    logic signed [36:0] sm_b;
    assign sm_a = $signed({1'b0, rd[17:0]});
    assign sm_b = $signed({1'b0, r});
    mul_pipe #(.AW(19), .BW(37), .TW(1)) u_mul (.clk, .en(1'b1), .vi(v1), .ti(t1),
        .a(sm_a), .b(sm_b), .vo(mvo), .to(mto), .p(mp));
    vcredit #(.D(FD)) u_c (.clk, .rst_n, .inc((st == OUT) && issue), .dec(v6), .f_cnt, .ok);
    vfifo #(.W(19), .D(FD)) u_f (.clk, .rst_n, .push(v6), .din({t6, p6}), .pop,
                                  .dout(f_dout), .empty(f_empty), .cnt(f_cnt));

    always_ff @(posedge clk) begin
        rc_start <= 1'b0;
        if (!rst_n) begin
            st <= LOAD; cnt <= '0; n <= '0; iss <= '0; wcnt <= '0; total <= '0; mx <= '0;
            l1v <= 1'b0; ei <= 1'b0; v1 <= 1'b0; v6 <= 1'b0;
        end else begin
            l1v <= (st == LOAD) && a_tvalid; l1x <= a_tdata; l1l <= a_tlast;
            ei <= (st == EXP) && issue;
            v1 <= (st == OUT) && issue; t1 <= (iss == n - 1'b1);
            v6 <= mvo; t6 <= mto; p6 <= 18'((mp + 56'sd2147483648) >>> 32);   // p = rsr(e·r, 32)
            case (st)
                LOAD: if (l1v) begin
                    if (cnt == 0 || l1x > mx) mx <= l1x;
                    cnt <= cnt + 1'b1;
                    if (l1l) begin n <= cnt + 1'b1; st <= EXP; iss <= '0; wcnt <= '0; total <= '0; end
                end
                EXP: begin
                    if (issue) iss <= iss + 1'b1;
                    if (eo) begin
                        total <= total + 40'(e);
                        wcnt <= wcnt + 1'b1;
                        if (wcnt == n - 1'b1) begin rc_start <= 1'b1; st <= SFU; end
                    end
                end
                SFU: if (rc_done) begin r <= rc_y; iss <= '0; st <= OUT; end
                OUT: begin
                    if (issue) iss <= iss + 1'b1;
                    if (pop && f_dout[18]) begin st <= LOAD; cnt <= '0; end
                end
                default: st <= LOAD;
            endcase
        end
    end
endmodule

// ============================================================ RMSNorm
module vpu_rmsnorm #(parameter int DEPTH = 2048) (
    input  logic        clk, rst_n,
    input  logic [24:0] recip_n,        // round(2^24 / n)，按维度预先算好的常数
    input  logic [63:0] a_tdata,        // x（Q16，int32）
    input  logic        a_tvalid, a_tlast,
    output logic        a_tready,
    input  logic [63:0] b_tdata,        // g（Q16，int32），输出阶段逐个消耗
    input  logic        b_tvalid, b_tlast,
    output logic        b_tready,
    output logic [63:0] y_tdata,
    output logic        y_tvalid, y_tlast,
    input  logic        y_tready
);
    `include "vpu_common.svh"
    localparam int AW = $clog2(DEPTH);
    localparam int FD = 16;
    typedef enum logic [2:0] {LOAD, SQW, MW, MS, SFU, OUT} st_t;
    st_t st;

    logic signed [31:0] mem [DEPTH];
    logic signed [31:0] rd;
    logic [AW-1:0]      ra;
    logic [AW:0]        cnt, n, iss, sqc;
    logic               l1v, l1l, l2v, sqv;
    logic signed [31:0] l1x;
    logic signed [23:0] xs2;
    logic signed [47:0] sq;
    logic [57:0]        ss;
    logic               mi, mo;
    logic signed [83:0] mprod;
    logic [83:0]        mh;
    logic [63:0]        mean;
    logic               rs_start, rs_done;
    logic [31:0]        rs_y, r;
    logic               ok1, issue, v1, t1, m1o, m1t, v6, t6, e1, pop1;
    logic signed [58:0] m1p;
    logic signed [41:0] y1_6;
    logic [$clog2(FD):0] f1_cnt;
    logic [42:0]        f1_dout;
    logic               ok2, fire2, m2o, m2t, v7, t7, v8, t8, e2, pop2;
    logic signed [73:0] m2p;
    logic signed [57:0] r7;
    logic signed [31:0] y8;
    logic [$clog2(FD):0] f2_cnt;
    logic [32:0]        f2_dout;

    assign a_tready = (st == LOAD);
    assign issue = (st == OUT) && (iss < n) && ok1;
    assign ra = iss[AW-1:0];
    assign fire2 = (st == OUT) && !e1 && b_tvalid && ok2;
    assign b_tready = (st == OUT) && !e1 && ok2;
    assign pop1 = fire2;
    assign y_tvalid = (st == OUT) && !e2;
    assign pop2 = y_tvalid && y_tready;
    assign y_tdata = 64'($signed(f2_dout[31:0]));
    assign y_tlast = f2_dout[32];

    always_ff @(posedge clk) begin
        if (l1v) mem[cnt[AW-1:0]] <= l1x;
        rd <= mem[ra];
    end

    mul_pipe #(.AW(24), .BW(24)) u_sq (.clk, .en(1'b1), .vi(l2v), .ti(1'b0), .a(xs2), .b(xs2),
                                       .vo(sqv), .to(), .p(sq));
    logic signed [58:0] mn_a;
    logic signed [24:0] mn_b;
    assign mn_a = $signed({1'b0, ss});
    assign mn_b = $signed(recip_n);
    mul_pipe #(.AW(59), .BW(25)) u_mean (.clk, .en(1'b1), .vi(mi), .ti(1'b0),
        .a(mn_a), .b(mn_b), .vo(mo), .to(), .p(mprod));
    vpu_rsqrt u_rsqrt (.clk, .rst_n, .start(rs_start), .m(mean), .done(rs_done), .y(rs_y));
    logic signed [26:0] m1_b;
    assign m1_b = $signed({1'b0, r[25:0]});
    mul_pipe #(.AW(32), .BW(27), .TW(1)) u_m1 (.clk, .en(1'b1), .vi(v1), .ti(t1),
        .a(rd), .b(m1_b), .vo(m1o), .to(m1t), .p(m1p));
    vcredit #(.D(FD)) u_c1 (.clk, .rst_n, .inc(issue), .dec(v6), .f_cnt(f1_cnt), .ok(ok1));
    vfifo #(.W(43), .D(FD)) u_f1 (.clk, .rst_n, .push(v6), .din({t6, y1_6}), .pop(pop1),
                                   .dout(f1_dout), .empty(e1), .cnt(f1_cnt));
    logic signed [41:0] m2_a;
    logic signed [31:0] m2_b;
    assign m2_a = $signed(f1_dout[41:0]);
    assign m2_b = $signed(b_tdata[31:0]);
    mul_pipe #(.AW(42), .BW(32), .TW(1)) u_m2 (.clk, .en(1'b1), .vi(fire2), .ti(f1_dout[42]),
        .a(m2_a), .b(m2_b), .vo(m2o), .to(m2t), .p(m2p));
    vcredit #(.D(FD)) u_c2 (.clk, .rst_n, .inc(fire2), .dec(v8), .f_cnt(f2_cnt), .ok(ok2));
    vfifo #(.W(33), .D(FD)) u_f2 (.clk, .rst_n, .push(v8), .din({t8, y8}), .pop(pop2),
                                   .dout(f2_dout), .empty(e2), .cnt(f2_cnt));

    always_ff @(posedge clk) begin
        rs_start <= 1'b0;
        mi <= 1'b0;
        if (!rst_n) begin
            st <= LOAD; cnt <= '0; n <= '0; ss <= '0; iss <= '0; sqc <= '0;
            l1v <= 1'b0; l2v <= 1'b0; v1 <= 1'b0; v6 <= 1'b0; v7 <= 1'b0; v8 <= 1'b0;
        end else begin
            // 平方和：xs = rsr(x, 8)，ss += xs²
            l1v <= (st == LOAD) && a_tvalid; l1x <= a_tdata[31:0]; l1l <= a_tlast;
            l2v <= l1v; xs2 <= 24'((33'(l1x) + 33'sd128) >>> 8);
            if (sqv) begin ss <= ss + 58'(sq); sqc <= sqc + 1'b1; end
            if (l1v) cnt <= cnt + 1'b1;
            // 第 1 段：y1 = rsr(x·r, 16)
            v1 <= issue; t1 <= (iss == n - 1'b1);
            v6 <= m1o; t6 <= m1t; y1_6 <= 42'((m1p + 59'sd32768) >>> 16);
            // 第 2 段：y = sat32(rsr(y1·g, 16))
            v7 <= m2o; t7 <= m2t; r7 <= 58'((m2p + 74'sd32768) >>> 16);
            v8 <= v7; t8 <= t7; y8 <= sat32(80'(r7));
            case (st)
                LOAD: if (l1v && l1l) begin n <= cnt + 1'b1; st <= SQW; end
                SQW: if (sqc == n) begin mi <= 1'b1; st <= MW; end   // 等平方和流水线排空
                MW: if (mo) begin mh <= 84'(mprod) + 84'd8388608; st <= MS; end
                MS: begin
                    mean <= 64'(mh >> 24) + 64'(EPS_Q16);            // rsr(ss·recip_n, 24) + eps
                    rs_start <= 1'b1;
                    st <= SFU;
                end
                SFU: if (rs_done) begin r <= rs_y; iss <= '0; st <= OUT; end
                OUT: begin
                    if (issue) iss <= iss + 1'b1;
                    if (pop2 && f2_dout[32]) begin st <= LOAD; cnt <= '0; ss <= '0; sqc <= '0; end
                end
                default: st <= LOAD;
            endcase
        end
    end
    wire _unused = &{1'b0, b_tlast, a_tdata[63:32], b_tdata[63:32]};
endmodule

// ============================================================ 逐元素：残差相加 / SwiGLU 门控
module vpu_ew (
    input  logic        clk, rst_n,
    input  logic        op,             // 0: sat32(a+b)  1: sat32(rsr(silu(a)·b, 16))
    input  logic [63:0] a_tdata,
    input  logic        a_tvalid, a_tlast,
    output logic        a_tready,
    input  logic [63:0] b_tdata,
    input  logic        b_tvalid, b_tlast,
    output logic        b_tready,
    output logic [63:0] y_tdata,
    output logic        y_tvalid, y_tlast,
    input  logic        y_tready
);
    `include "vpu_common.svh"
    localparam int FD = 32;
    localparam int TW = 1 + 1 + 32 + 32;    // {op, last, resadd 结果, h}
    logic               ok, fire, pop, f_empty;
    logic [$clog2(FD):0] f_cnt;
    logic [3:0]         v;
    logic [TW-1:0]      tg [4];
    logic signed [31:0] x [4];
    logic signed [32:0] pos;
    logic [9:0]         idx1;
    logic [12:0]        fr1, fr2;
    logic [16:0]        la2, lb2, la3;
    logic [29:0]        dm3;
    logic [17:0]        sig4;
    logic               m1o, v5;
    logic [TW-1:0]      m1t, tg5;
    logic signed [50:0] m1p;
    logic signed [33:0] silu5;
    logic               m2o, v7, v8;
    logic [33:0]        m2t, tg7, tg8;
    logic signed [65:0] m2p;
    logic signed [49:0] r7;
    logic signed [31:0] y8;
    logic [32:0]        f_dout;

    assign fire = a_tvalid && b_tvalid && ok;
    assign a_tready = b_tvalid && ok;
    assign b_tready = a_tvalid && ok;
    assign y_tvalid = !f_empty;
    assign pop = y_tvalid && y_tready;
    assign y_tdata = 64'($signed(f_dout[31:0]));
    assign y_tlast = f_dout[32];

    always_comb begin
        pos = 33'(x[0]) + 33'sd1048576;                       // x + 16·2^16
        if (pos < 0) pos = 0;
        if (pos > 33'sd2097152) pos = 33'sd2097152;            // 512·2^12
        idx1 = (pos[32:12] > 21'd511) ? 10'd511 : 10'(pos[32:12]);
        fr1 = 13'(pos - 33'($signed({1'b0, idx1, 12'd0})));
    end

    logic signed [18:0] e1_b;
    assign e1_b = $signed({1'b0, sig4});
    mul_pipe #(.AW(32), .BW(19), .TW(TW)) u_m1 (.clk, .en(1'b1), .vi(v[3]), .ti(tg[3]),
        .a(x[3]), .b(e1_b), .vo(m1o), .to(m1t), .p(m1p));
    logic signed [31:0] e2_b;
    assign e2_b = $signed(tg5[31:0]);
    mul_pipe #(.AW(34), .BW(32), .TW(34)) u_m2 (.clk, .en(1'b1), .vi(v5), .ti(tg5[TW-1:32]),
        .a(silu5), .b(e2_b), .vo(m2o), .to(m2t), .p(m2p));
    vcredit #(.D(FD)) u_c (.clk, .rst_n, .inc(fire), .dec(v8), .f_cnt, .ok);
    vfifo #(.W(33), .D(FD)) u_f (.clk, .rst_n, .push(v8), .din({tg8[32], y8}), .pop,
                                  .dout(f_dout), .empty(f_empty), .cnt(f_cnt));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v <= '0; v5 <= 1'b0; v7 <= 1'b0; v8 <= 1'b0;
        end else begin
            v <= {v[2:0], fire};
            v5 <= m1o; v7 <= m2o; v8 <= v7;
        end
        // 0：寄存输入（tg = {op, last, 预留, h}）
        x[0] <= $signed(a_tdata[31:0]);
        tg[0] <= {op, a_tlast, 32'd0, b_tdata[31:0]};
        // 1：查表地址（组合）+ 查表；顺便算残差相加
        x[1] <= x[0];
        tg[1] <= {tg[0][TW-1:TW-2], sat32(80'(x[0]) + 80'($signed(tg[0][31:0]))), tg[0][31:0]};
        fr2 <= fr1; la2 <= sig_lut(idx1); lb2 <= sig_lut(idx1 + 10'd1);
        // 2：插值乘
        x[2] <= x[1]; tg[2] <= tg[1];
        dm3 <= 30'(lb2 - la2) * 30'(fr2); la3 <= la2;
        // 3：插值加
        x[3] <= x[2]; tg[3] <= tg[2];
        sig4 <= 18'(la3) + 18'(dm3 >> 12);
        // silu = rsr(x·sig, 16)
        tg5 <= m1t; silu5 <= 34'((m1p + 51'sd32768) >>> 16);
        // y = op ? sat32(rsr(silu·h, 16)) : resadd
        tg7 <= m2t; r7 <= 50'((m2p + 66'sd32768) >>> 16);
        tg8 <= tg7; y8 <= tg7[33] ? sat32(80'(r7)) : $signed(tg7[31:0]);
    end
    wire _unused = &{1'b0, b_tlast, a_tdata[63:32], b_tdata[63:32]};
endmodule

// ============================================================ RoPE
module vpu_rope (
    input  logic        clk, rst_n,
    input  logic [63:0] a_tdata,        // v0, v1, v0, v1, ...（int32）
    input  logic        a_tvalid, a_tlast,
    output logic        a_tready,
    input  logic [63:0] b_tdata,        // 每对一个 {sin, cos}（Q14）
    input  logic        b_tvalid, b_tlast,
    output logic        b_tready,
    output logic [63:0] y_tdata,
    output logic        y_tvalid, y_tlast,
    input  logic        y_tready
);
    `include "vpu_common.svh"
    localparam int FD = 8;
    logic               half_in, half_out, ok, fire1, pop, f_empty;
    logic signed [31:0] v0, av0, av1;
    logic signed [15:0] ac, as_;
    logic               av, al;
    logic               mo, ml, s1v, s1l, s2v, s2l;
    logic signed [47:0] p00, p11, p01, p10;
    logic signed [48:0] d0, d1;
    logic signed [34:0] o0, o1;
    logic [$clog2(FD):0] f_cnt;
    logic [70:0]        f_dout;

    assign fire1 = half_in && a_tvalid && b_tvalid && ok;      // 收到 v1 与 {sin,cos}
    assign a_tready = !half_in || (b_tvalid && ok);
    assign b_tready = half_in && a_tvalid && ok;
    assign y_tvalid = !f_empty;
    assign y_tdata  = half_out ? 64'($signed(f_dout[69:35])) : 64'($signed(f_dout[34:0]));
    assign y_tlast  = half_out && f_dout[70];
    assign pop = y_tvalid && y_tready && half_out;

    mul_pipe #(.AW(32), .BW(16), .TW(1)) u_00 (.clk, .en(1'b1), .vi(av), .ti(al), .a(av0), .b(ac),  .vo(mo), .to(ml), .p(p00));
    mul_pipe #(.AW(32), .BW(16))         u_11 (.clk, .en(1'b1), .vi(av), .ti(1'b0), .a(av1), .b(as_), .vo(), .to(), .p(p11));
    mul_pipe #(.AW(32), .BW(16))         u_01 (.clk, .en(1'b1), .vi(av), .ti(1'b0), .a(av0), .b(as_), .vo(), .to(), .p(p01));
    mul_pipe #(.AW(32), .BW(16))         u_10 (.clk, .en(1'b1), .vi(av), .ti(1'b0), .a(av1), .b(ac),  .vo(), .to(), .p(p10));
    vcredit #(.D(FD)) u_c (.clk, .rst_n, .inc(fire1), .dec(s2v), .f_cnt, .ok);
    vfifo #(.W(71), .D(FD)) u_f (.clk, .rst_n, .push(s2v), .din({s2l, o1, o0}), .pop,
                                  .dout(f_dout), .empty(f_empty), .cnt(f_cnt));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            half_in <= 1'b0; half_out <= 1'b0; av <= 1'b0; s1v <= 1'b0; s2v <= 1'b0;
        end else begin
            if (!half_in && a_tvalid) begin v0 <= $signed(a_tdata[31:0]); half_in <= 1'b1; end
            if (fire1) half_in <= 1'b0;
            av <= fire1;
            s1v <= mo; s2v <= s1v;
            if (y_tvalid && y_tready) half_out <= !half_out;
        end
        av0 <= v0; av1 <= $signed(a_tdata[31:0]); al <= a_tlast;
        ac <= $signed(b_tdata[15:0]); as_ <= $signed(b_tdata[47:32]);
        s1l <= ml; d0 <= 49'(p00) - 49'(p11); d1 <= 49'(p01) + 49'(p10);
        s2l <= s1l; o0 <= 35'((d0 + 49'sd8192) >>> 14); o1 <= 35'((d1 + 49'sd8192) >>> 14);
    end
    wire _unused = &{1'b0, b_tlast, a_tdata[63:32], b_tdata[31:16], b_tdata[63:48]};
endmodule

// ============================================================ 注意力（一个头）
// 输入行流 r：每拍一行 = HSMAX 个 int8（字节 0..HSMAX-1，hs 之后补零）+ 字节 HSMAX 的移位量。
// 顺序：q 行（含 qs）→ T 个 K 行（含 ks，最后一行 tlast）→ T 个 V 行（含 vs，最后一行 tlast）。
// 每行分 HSMAX/LANES 拍处理。输出 hs 个 Q16 结果（64 位）。
module attn_head #(parameter int HSMAX = 64, parameter int LANES = 16, parameter int DEPTH = 2048) (
    input  logic                   clk, rst_n,
    input  logic [16:0]            inv_sqrt_hs,   // round(2^16/√hs)
    input  logic [7:0]             hs,
    input  logic [(HSMAX+1)*8-1:0] r_tdata,
    input  logic                   r_tvalid, r_tlast,
    output logic                   r_tready,
    output logic [63:0]            y_tdata,
    output logic                   y_tvalid, y_tlast,
    input  logic                   y_tready
);
    `include "vpu_common.svh"
    localparam int AW  = $clog2(DEPTH);
    localparam int SUB = HSMAX / LANES;
    localparam int SBW = $clog2(SUB);
    localparam int FD  = 8;
    typedef enum logic [3:0] {QLOAD, KLOAD, EXP, SFU, PCALC, VPRE, VLOAD, VDRAIN, OUT} st_t;
    st_t st;

    logic [HSMAX*8-1:0] q;
    logic [7:0]         qs, rs;
    logic [SBW-1:0]     sub;
    logic               kin_done;
    logic signed [63:0] mem [DEPTH];           // 分数 → e → p 原地复用
    logic signed [63:0] rd, mx;
    logic [AW-1:0]      ra, wa;
    logic               we;
    logic [63:0]        wd;
    logic [AW:0]        T, wcnt, iss;
    // q·K：选通道 → 乘 → 4 路和 → 行内部分和 → 行累加 → ×inv → 加半 LSB → 移位
    logic               ka_v, ka_first, ka_end, ka_last;
    logic [7:0]         ka_ks;
    logic signed [7:0]  ka_q [LANES], ka_k [LANES];
    logic               kb_v, kb_first, kb_end, kb_last;
    logic [7:0]         kb_ks;
    logic signed [15:0] kb_p [LANES];
    logic               kc_v, kc_first, kc_end, kc_last;
    logic [7:0]         kc_ks;
    logic signed [17:0] kc_g [LANES/4];
    logic signed [19:0] kc_sum;
    logic               kd_v, kd_first, kd_end, kd_last;
    logic [7:0]         kd_ks;
    logic signed [19:0] kd_s;
    logic signed [20:0] dacc, dsum;
    logic               ke_v, ke_last;
    logic [7:0]         ke_ks;
    logic signed [20:0] ke_dot;
    logic               kf_v, kf_last;
    logic signed [7:0]  kf_sh;
    logic signed [37:0] kf_m;
    logic               kg_v, kg_last;
    logic signed [7:0]  kg_sh;
    logic signed [63:0] kg_m;
    logic               d_v, d_last;
    logic signed [63:0] d_score;
    // 指数 / 倒数 / 概率
    logic               ei, eo;
    logic [17:0]        e;
    logic [39:0]        total;
    logic               rc_start, rc_done;
    logic [35:0]        rc_y, rr;
    logic               pv1, pmo, pv6;
    logic [AW:0]        pidx1, pmt, pidx6;
    logic signed [55:0] pmp;
    logic [17:0]        p6;
    // P·V：选通道 → 乘 → 移位 → 累加
    logic [AW:0]        vt;
    logic signed [59:0] acc [HSMAX];
    logic               va_v, vb_v, vc_v;
    logic [SBW-1:0]     va_sub, vb_sub, vc_sub;
    logic [7:0]         va_rs, vb_rs;
    logic [17:0]        va_p;
    logic signed [7:0]  va_v8 [LANES];
    logic signed [26:0] vb_pr [LANES];
    logic signed [59:0] vc_sh [LANES];
    // 输出
    logic               ok, oissue, ov1, ot1, ov2, ot2, pop, f_empty;
    logic [7:0]         oidx;
    logic signed [59:0] oa1;
    logic signed [63:0] oy2;
    logic [$clog2(FD):0] f_cnt;
    logic [64:0]        f_dout;

    assign rs = r_tdata[HSMAX*8 +: 8];
    assign r_tready = (st == QLOAD) || ((st == KLOAD) && !kin_done && sub == SBW'(SUB - 1))
                   || ((st == VLOAD) && sub == SBW'(SUB - 1));
    assign dsum = (kd_first ? 21'sd0 : dacc) + 21'(kd_s);
    assign oissue = (st == OUT) && (oidx < hs) && ok;
    assign y_tvalid = (st == OUT) && !f_empty;
    assign pop = y_tvalid && y_tready;
    assign y_tdata = f_dout[63:0];
    assign y_tlast = f_dout[64];

    always_comb begin
        kc_sum = '0;
        for (int g = 0; g < LANES / 4; g++) kc_sum = kc_sum + 20'(kc_g[g]);
    end

    // 读地址：EXP / PCALC 顺序读；VLOAD 时在行末预取下一行的 p
    always_comb begin
        ra = iss[AW-1:0];
        if (st == VPRE) ra = '0;
        else if (st == VLOAD) ra = (r_tvalid && sub == SBW'(SUB - 1)) ? AW'(vt + 1'b1) : vt[AW-1:0];
        we = 1'b0; wa = wcnt[AW-1:0]; wd = d_score;
        if (st == KLOAD && d_v) we = 1'b1;
        else if (eo) begin we = 1'b1; wd = 64'(e); end
        else if (pv6) begin we = 1'b1; wa = pidx6[AW-1:0]; wd = 64'(p6); end
    end
    always_ff @(posedge clk) begin
        if (we) mem[wa] <= wd;
        rd <= mem[ra];
    end

    vpu_exp_pipe u_exp (.clk, .vi(ei), .x(rd), .mx, .vo(eo), .e);
    vpu_recip u_recip (.clk, .rst_n, .start(rc_start), .m(total), .done(rc_done), .y(rc_y));
    logic signed [18:0] pm_a;
    logic signed [36:0] pm_b;
    assign pm_a = $signed({1'b0, rd[17:0]});
    assign pm_b = $signed({1'b0, rr});
    mul_pipe #(.AW(19), .BW(37), .TW(AW+1)) u_pm (.clk, .en(1'b1), .vi(pv1), .ti(pidx1),
        .a(pm_a), .b(pm_b), .vo(pmo), .to(pmt), .p(pmp));
    vcredit #(.D(FD)) u_c (.clk, .rst_n, .inc(oissue), .dec(ov2), .f_cnt, .ok);
    vfifo #(.W(65), .D(FD)) u_f (.clk, .rst_n, .push(ov2), .din({ot2, oy2}), .pop,
                                  .dout(f_dout), .empty(f_empty), .cnt(f_cnt));

    always_ff @(posedge clk) begin
        rc_start <= 1'b0;
        if (!rst_n) begin
            st <= QLOAD; sub <= '0; kin_done <= 1'b0;
            ka_v <= 1'b0; kb_v <= 1'b0; kc_v <= 1'b0; kd_v <= 1'b0; ke_v <= 1'b0; kf_v <= 1'b0;
            kg_v <= 1'b0; d_v <= 1'b0; ei <= 1'b0; pv1 <= 1'b0; pv6 <= 1'b0;
            va_v <= 1'b0; vb_v <= 1'b0; vc_v <= 1'b0; ov1 <= 1'b0; ov2 <= 1'b0;
            wcnt <= '0; iss <= '0; T <= '0; vt <= '0; oidx <= '0; total <= '0; mx <= '0;
        end else begin
            // ---- q·K
            ka_v <= (st == KLOAD) && !kin_done && r_tvalid;
            ka_first <= (sub == 0); ka_end <= (sub == SBW'(SUB - 1)); ka_last <= r_tlast; ka_ks <= rs;
            for (int l = 0; l < LANES; l++) begin
                ka_q[l] <= $signed(q[8*(int'(sub)*LANES + l) +: 8]);
                ka_k[l] <= $signed(r_tdata[8*(int'(sub)*LANES + l) +: 8]);
            end
            kb_v <= ka_v; kb_first <= ka_first; kb_end <= ka_end; kb_last <= ka_last; kb_ks <= ka_ks;
            for (int l = 0; l < LANES; l++) kb_p[l] <= 16'(ka_q[l]) * 16'(ka_k[l]);
            kc_v <= kb_v; kc_first <= kb_first; kc_end <= kb_end; kc_last <= kb_last; kc_ks <= kb_ks;
            for (int g = 0; g < LANES / 4; g++)
                kc_g[g] <= 18'(kb_p[4*g]) + 18'(kb_p[4*g+1]) + 18'(kb_p[4*g+2]) + 18'(kb_p[4*g+3]);
            kd_v <= kc_v; kd_first <= kc_first; kd_end <= kc_end; kd_last <= kc_last; kd_ks <= kc_ks;
            kd_s <= kc_sum;
            if (kd_v) dacc <= dsum;
            ke_v <= kd_v && kd_end; ke_dot <= dsum; ke_ks <= kd_ks; ke_last <= kd_last;
            kf_v <= ke_v; kf_last <= ke_last;
            kf_m <= 38'(ke_dot) * 38'($signed({1'b0, inv_sqrt_hs}));
            kf_sh <= 8'sd32 - $signed(qs) - $signed(ke_ks);
            kg_v <= kf_v; kg_last <= kf_last; kg_sh <= kf_sh;
            kg_m <= 64'(kf_m) + ((kf_sh > 0) ? (64'sd1 <<< (kf_sh - 8'sd1)) : 64'sd0);
            d_v <= kg_v; d_last <= kg_last;
            d_score <= (kg_sh > 0) ? (kg_m >>> kg_sh) : (kg_m <<< (-kg_sh));
            // ---- 指数（读 1 拍后进指数流水线）
            ei <= (st == EXP) && (iss < T);
            // ---- 概率：p = rsr(e·r, 32)，写回
            pv1 <= (st == PCALC) && (iss < T); pidx1 <= iss;
            pv6 <= pmo; pidx6 <= pmt; p6 <= 18'((pmp + 56'sd2147483648) >>> 32);
            // ---- P·V
            va_v <= (st == VLOAD) && r_tvalid; va_sub <= sub; va_rs <= rs; va_p <= rd[17:0];
            for (int l = 0; l < LANES; l++) va_v8[l] <= $signed(r_tdata[8*(int'(sub)*LANES + l) +: 8]);
            vb_v <= va_v; vb_sub <= va_sub; vb_rs <= va_rs;
            for (int l = 0; l < LANES; l++) vb_pr[l] <= 27'($signed({1'b0, va_p})) * 27'(va_v8[l]);
            vc_v <= vb_v; vc_sub <= vb_sub;
            for (int l = 0; l < LANES; l++) vc_sh[l] <= 60'(vb_pr[l]) <<< vb_rs;
            if (vc_v)
                for (int l = 0; l < LANES; l++)
                    acc[int'(vc_sub)*LANES + l] <= acc[int'(vc_sub)*LANES + l] + vc_sh[l];
            // ---- 输出：读累加器 → rsr(·, 16)
            ov1 <= oissue; ot1 <= (oidx == hs - 1'b1); oa1 <= acc[oidx[5:0]];
            ov2 <= ov1; ot2 <= ot1; oy2 <= (64'(oa1) + 64'sd32768) >>> 16;

            case (st)
                QLOAD: if (r_tvalid) begin
                    q <= r_tdata[HSMAX*8-1:0];
                    qs <= rs;
                    for (int j = 0; j < HSMAX; j++) acc[j] <= '0;
                    sub <= '0; wcnt <= '0; kin_done <= 1'b0;
                    st <= KLOAD;
                end
                KLOAD: begin
                    if (!kin_done && r_tvalid) begin
                        sub <= sub + 1'b1;
                        if (sub == SBW'(SUB - 1) && r_tlast) kin_done <= 1'b1;
                    end
                    if (d_v) begin
                        if (wcnt == 0 || d_score > mx) mx <= d_score;
                        wcnt <= wcnt + 1'b1;
                        if (d_last) begin T <= wcnt + 1'b1; iss <= '0; total <= '0; st <= EXP; end
                    end
                end
                EXP: begin
                    if (iss < T) iss <= iss + 1'b1;
                    if (iss == 0) wcnt <= '0;
                    if (eo) begin
                        total <= total + 40'(e);
                        wcnt <= wcnt + 1'b1;
                        if (wcnt == T - 1'b1) begin rc_start <= 1'b1; st <= SFU; end
                    end
                end
                SFU: if (rc_done) begin rr <= rc_y; iss <= '0; st <= PCALC; end
                PCALC: begin
                    if (iss < T) iss <= iss + 1'b1;
                    if (pv6 && pidx6 == T - 1'b1) st <= VPRE;
                end
                VPRE: begin vt <= '0; sub <= '0; st <= VLOAD; end
                VLOAD: if (r_tvalid) begin
                    sub <= sub + 1'b1;
                    if (sub == SBW'(SUB - 1)) begin
                        vt <= vt + 1'b1;
                        if (r_tlast) st <= VDRAIN;
                    end
                end
                VDRAIN: if (!va_v && !vb_v && !vc_v) begin oidx <= '0; st <= OUT; end
                OUT: begin
                    if (oissue) oidx <= oidx + 1'b1;
                    if (pop && f_dout[64]) st <= QLOAD;
                end
                default: st <= QLOAD;
            endcase
        end
    end
endmodule
