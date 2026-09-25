// RETIMED against sec_gold.v, and genuinely equivalent: the `+1` has moved
// BACKWARD across the register and the reset value has been adjusted from 0 to
// 1 to compensate. Same latency, same output on every cycle after reset --
// confirmed independently by 2000 cycles of Verilator simulation with random
// stimulus (0 mismatches) as well as by kepler.
//
// The register now holds a DIFFERENT value than its counterpart in the gold
// design, which is exactly what a retiming pass produces and exactly what a
// partition-based checker cannot accept. Measured on this pair:
//
//   eqy            refuted in 0.66s   "partitions not equivalent: rt.r, rt.q"
//   kepler-formal  proven  in 0.06s   SEC, 100% output coverage, k = 1
//
// eqy is not misbehaving -- gate-level LEC documents that it requires unchanged
// sequential boundaries -- but its answer here is a FALSE refutation, and a gate
// running eqy alone rejects this whole class of valid edit.
module rt(input clk, input resetn, input [7:0] d, output [7:0] q);
  reg [7:0] r;
  always @(posedge clk) begin
    if (!resetn) r <= 8'd1;
    else         r <= d + 8'd1;
  end
  assign q = r;
endmodule
