// Textually different, functionally identical: the operands are commuted and
// the sum is hoisted into a wire. A gate that cannot prove THIS equivalent is
// useless -- it would reject every harmless refactor an agent proposes.
module tiny(input clk, input [3:0] a, input [3:0] b, output reg [3:0] y);
  wire [3:0] s = b + a;
  always @(posedge clk) y <= s;
endmodule
