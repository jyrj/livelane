// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

module ibex_counter #(
  parameter int CounterWidth = 32,
  // When set `counter_val_upd_o` provides an incremented version of the counter value, otherwise
  // the output is hard-wired to 0. This is required to allow Xilinx DSP inference to work
  // correctly. When `ProvideValUpd` is set no DSPs are inferred.
  parameter bit ProvideValUpd = 0
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic        counter_inc_i,
  input  logic        counterh_we_i,
  input  logic        counter_we_i,
  input  logic [31:0] counter_val_i,
  output logic [63:0] counter_val_o,
  output logic [63:0] counter_val_upd_o
);

  logic [63:0]             counter;
  logic [CounterWidth-1:0] counter_upd;
  logic [63:0]             counter_load;
  logic                    we;
  logic [CounterWidth-1:0] counter_d;

  // Increment
  localparam int NumBlocks = (CounterWidth + 3) / 4;
  logic [NumBlocks-1:0]    block_prop;
  logic [NumBlocks-1:0]    block_carry;
  logic [CounterWidth-1:0] counter_inc_carry;

  for (genvar b = 0; b < NumBlocks; b++) begin : g_block_prop
    localparam int TopBit = (b * 4 + 3 < CounterWidth) ? (b * 4 + 3) : (CounterWidth - 1);
    localparam int BotBit = b * 4;
    assign block_prop[b] = &counter[TopBit:BotBit];
  end

  assign block_carry[0] = 1'b1;
  for (genvar b = 1; b < NumBlocks; b++) begin : g_block_carry
    assign block_carry[b] = &block_prop[b-1:0];
  end

  for (genvar g = 0; g < CounterWidth; g++) begin : g_counter_inc
    localparam int BlockIdx = g / 4;
    localparam int RemIdx   = g % 4;
    if (RemIdx == 0) begin : g_rem_zero
      assign counter_inc_carry[g] = block_carry[BlockIdx];
    end else begin : g_rem_nonzero
      assign counter_inc_carry[g] = block_carry[BlockIdx] & (&counter[g-1 : BlockIdx*4]);
    end
    assign counter_upd[g] = counter[g] ^ counter_inc_carry[g];
  end

  // Update
  always_comb begin
    // Write
    we = counter_we_i | counterh_we_i;
    counter_load[63:32] = counter[63:32];
    counter_load[31:0]  = counter_val_i;
    if (counterh_we_i) begin
      counter_load[63:32] = counter_val_i;
      counter_load[31:0]  = counter[31:0];
    end

    // Next value logic
    if (we) begin
      counter_d = counter_load[CounterWidth-1:0];
    end else if (counter_inc_i) begin
      counter_d = counter_upd[CounterWidth-1:0];
    end else begin
      counter_d = counter[CounterWidth-1:0];
    end
  end

`ifdef FPGA_XILINX
  // On Xilinx FPGAs, 48-bit DSPs are available that can be used for the
  // counter. Hence, use Xilinx specific flop implementation. The datatype for
  // UseDsp is on purpose int as with string Xilinx throws an error for the
  // use_dsp pragma.
  localparam int UseDsp = CounterWidth < 49 ? "yes" : "no";
  (* use_dsp = UseDsp *) logic [CounterWidth-1:0] counter_q;
`else
  localparam int UseDsp = "no";
  logic [CounterWidth-1:0] counter_q;
`endif

  if (UseDsp == "yes") begin : g_cnt_dsp
    // Use sync. reset for DSP.
    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        counter_q <= '0;
      end else begin
        counter_q <= counter_d;
      end
    end
  end else begin : g_cnt_no_dsp
    // Use async. reset for flop.
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        counter_q <= '0;
      end else begin
        counter_q <= counter_d;
      end
    end
  end


  if (CounterWidth < 64) begin : g_counter_narrow
    logic [63:CounterWidth] unused_counter_load;

    assign counter[CounterWidth-1:0]           = counter_q;
    assign counter[63:CounterWidth]            = '0;

    if (ProvideValUpd) begin : g_counter_val_upd_o
      assign counter_val_upd_o[CounterWidth-1:0] = counter_upd;
    end else begin : g_no_counter_val_upd_o
      assign counter_val_upd_o[CounterWidth-1:0] = '0;
    end
    assign counter_val_upd_o[63:CounterWidth]  = '0;
    assign unused_counter_load                 = counter_load[63:CounterWidth];
  end else begin : g_counter_full
    assign counter           = counter_q;

    if (ProvideValUpd) begin : g_counter_val_upd_o
      assign counter_val_upd_o = counter_upd;
    end else begin : g_no_counter_val_upd_o
      assign counter_val_upd_o = '0;
    end
  end

  assign counter_val_o = counter;

endmodule
