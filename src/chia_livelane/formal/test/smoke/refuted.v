// A real functional bug: subtraction where the design adds. It elaborates,
// synthesises and simulates without complaint on any input where a >= b, so
// neither a compiler nor a short random test bench refutes it. Only an
// equivalence check does.
module tiny(input clk, input [3:0] a, input [3:0] b, output reg [3:0] y);
  always @(posedge clk) y <= a - b;
endmodule
