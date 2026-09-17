// The known-good design. Deliberately tiny: the point of the fixture is to
// prove that the checker can distinguish equivalence from non-equivalence,
// not to stress the solver.
module tiny(input clk, input [3:0] a, input [3:0] b, output reg [3:0] y);
  always @(posedge clk) y <= a + b;
endmodule
