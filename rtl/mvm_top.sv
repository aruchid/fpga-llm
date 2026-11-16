// MVM 加速器顶层：AXI4-Lite 寄存器 + x 缓冲，AXI-Stream 进权重、出结果。
// 上板时：AXI4-Lite 接 XDMA 的 BAR（M_AXI_LITE），两个流接 XDMA 的 H2C / C2H 流通道
// （或接从 DDR 读权重的 DMA）。仿真时由 cocotbext-axi 扮演这几个主从端。
//
// 寄存器（字节地址）：
//   0x000 CTRL   W  bit0=1 启动
//   0x004 STATUS R  bit0 busy，bit1 done（启动时清零）
//   0x008 KB     RW 输入维度块数 K/32
//   0x00C NT     RW 输出 tile 数 M/32
//   0x010 SHIFT  RW 24 - s（int8 补码）
//   0x014 CYCLES R  上次任务的周期数
//   0x018 ID     R  0x4D564D01（"MVM" v1）
//   0x1000-0x13FC  X  W  x 缓冲，int8 小端打包，字 i = x[4i..4i+3]，最多 1024 个元素
module mvm_top #(
    parameter int N = 32
) (
    input  logic            aclk,
    input  logic            aresetn,
    // AXI4-Lite 从端
    input  logic [15:0]     s_axil_awaddr,
    input  logic            s_axil_awvalid,
    output logic            s_axil_awready,
    input  logic [31:0]     s_axil_wdata,
    input  logic [3:0]      s_axil_wstrb,
    input  logic            s_axil_wvalid,
    output logic            s_axil_wready,
    output logic [1:0]      s_axil_bresp,
    output logic            s_axil_bvalid,
    input  logic            s_axil_bready,
    input  logic [15:0]     s_axil_araddr,
    input  logic            s_axil_arvalid,
    output logic            s_axil_arready,
    output logic [31:0]     s_axil_rdata,
    output logic [1:0]      s_axil_rresp,
    output logic            s_axil_rvalid,
    input  logic            s_axil_rready,
    // 权重流（H2C）
    input  logic [N*8-1:0]  s_axis_w_tdata,
    input  logic            s_axis_w_tvalid,
    output logic            s_axis_w_tready,
    // 结果流（C2H）
    output logic [31:0]     m_axis_y_tdata,
    output logic            m_axis_y_tvalid,
    input  logic            m_axis_y_tready,
    output logic            m_axis_y_tlast
);
    localparam int KBW   = 6;
    localparam int NTW   = 16;
    localparam int XBLKS = 32;            // x 缓冲块数（每块 N 个 int8）
    localparam int WPB   = N / 4;         // 每块的 32 位字数

    logic [KBW-1:0]     reg_kb;
    logic [NTW-1:0]     reg_nt;
    logic signed [7:0]  reg_shift;
    logic [31:0]        reg_cycles;
    logic               done_flag, start;
    logic               busy, done;
    logic [KBW-1:0]     x_idx;
    logic [N*8-1:0]     xmem [XBLKS];     // 同步读写的块 RAM（字节写使能）
    logic [N*8-1:0]     x_blk;

    // ---------------- AXI4-Lite 写：AW 与 W 同时到达才接收
    wire wr_fire = s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid;
    assign s_axil_awready = wr_fire;
    assign s_axil_wready  = wr_fire;
    assign s_axil_bresp   = 2'b00;

    wire [15:0] wa = s_axil_awaddr;
    wire        wr_x = wa[15:12] == 4'h1;
    wire [4:0]  wr_blk  = wa[9:5];
    wire [2:0]  wr_word = wa[4:2];

    always_ff @(posedge aclk) begin
        start <= 1'b0;
        if (!aresetn) begin
            s_axil_bvalid <= 1'b0;
            reg_kb <= KBW'(1); reg_nt <= NTW'(1); reg_shift <= '0;
        end else begin
            if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
            if (wr_fire) begin
                s_axil_bvalid <= 1'b1;
                if (wr_x) begin
                    for (int b = 0; b < 4; b++)
                        if (s_axil_wstrb[b])
                            xmem[wr_blk][32*int'(wr_word) + 8*b +: 8] <= s_axil_wdata[8*b +: 8];
                end else begin
                    case (wa[11:0])
                        12'h000: start     <= s_axil_wdata[0];
                        12'h008: reg_kb    <= s_axil_wdata[KBW-1:0];
                        12'h00C: reg_nt    <= s_axil_wdata[NTW-1:0];
                        12'h010: reg_shift <= s_axil_wdata[7:0];
                        default: ;
                    endcase
                end
            end
        end
    end

    always_ff @(posedge aclk) x_blk <= xmem[x_idx[4:0]];

    // ---------------- AXI4-Lite 读
    assign s_axil_arready = !s_axil_rvalid;
    assign s_axil_rresp   = 2'b00;
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axil_rvalid <= 1'b0;
            s_axil_rdata  <= '0;
        end else begin
            if (s_axil_rvalid && s_axil_rready) s_axil_rvalid <= 1'b0;
            if (s_axil_arvalid && s_axil_arready) begin
                s_axil_rvalid <= 1'b1;
                case (s_axil_araddr[11:0])
                    12'h004: s_axil_rdata <= {30'd0, done_flag, busy};
                    12'h008: s_axil_rdata <= 32'(reg_kb);
                    12'h00C: s_axil_rdata <= 32'(reg_nt);
                    12'h010: s_axil_rdata <= 32'(reg_shift);
                    12'h014: s_axil_rdata <= reg_cycles;
                    12'h018: s_axil_rdata <= 32'h4D56_4D01;
                    default: s_axil_rdata <= '0;
                endcase
            end
        end
    end

    // ---------------- 状态与性能计数
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            done_flag  <= 1'b0;
            reg_cycles <= '0;
        end else begin
            if (start) begin
                done_flag  <= 1'b0;
                reg_cycles <= '0;
            end else if (busy)
                reg_cycles <= reg_cycles + 1'b1;
            if (done) done_flag <= 1'b1;
        end
    end

    mvm_core #(.N(N), .KBW(KBW), .NTW(NTW)) u_core (
        .clk      (aclk),
        .rst_n    (aresetn),
        .start    (start),
        .kb_num   (reg_kb),
        .nt_num   (reg_nt),
        .shift    (reg_shift),
        .busy     (busy),
        .done     (done),
        .x_idx    (x_idx),
        .x_blk    (x_blk),
        .w_tdata  (s_axis_w_tdata),
        .w_tvalid (s_axis_w_tvalid),
        .w_tready (s_axis_w_tready),
        .y_tdata  (m_axis_y_tdata),
        .y_tvalid (m_axis_y_tvalid),
        .y_tready (m_axis_y_tready),
        .y_tlast  (m_axis_y_tlast)
    );

    // 地址解码假定每块 8 个字（N=32）
`ifndef SYNTHESIS
    initial assert (WPB == 8) else $error("x 缓冲地址解码假定 N=32");
`endif

    // 未用到的位（读地址不区分 x 区，x 缓冲只有 32 块）
    wire _unused = &{1'b0, x_idx[KBW-1], s_axil_awaddr[11:10], s_axil_araddr[15:12]};
endmodule
