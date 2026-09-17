// A functional bug that the FAST strategy gets WRONG, which is why it exists.
//
// The output is stuck at a constant: obviously not equivalent to `y <= a + b`.
// `sby`/smtbmc refutes it in 0.2s with a counterexample trace. Yosys's built-in
// `sat -tempinduct` -- eqy's `use sat` strategy -- reports "Induction step
// proven: SUCCESS!" and eqy turns that into PASS, so a ladder that lists
// `induct` before `smt` ADMITS this edit.
//
// The sibling fixture refuted.v (`a - b`) does NOT expose that: `sat` answers
// "unknown" there and the ladder falls through to `smt`, which refutes. So this
// file is the one that pins the ordering of DEFAULT_STRATEGIES.
module tiny(input clk, input [3:0] a, input [3:0] b, output reg [3:0] y);
  always @(posedge clk) y <= 4'b0;
endmodule
