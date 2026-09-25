// Known-good design for the SEQUENTIAL-equivalence fixture pair.
//
// Kept separate from gold.v because the discriminating ingredient is the
// RESET: sec_retimed.v moves logic across the register and adjusts the reset
// value to match, and without a reset both designs optimise back to the same
// thing and neither backend is tested on anything interesting.
module rt(input clk, input resetn, input [7:0] d, output [7:0] q);
  reg [7:0] r;
  always @(posedge clk) begin
    if (!resetn) r <= 8'd0;
    else         r <= d;
  end
  assign q = r + 8'd1;
endmodule
