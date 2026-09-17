// Smallest design that exercises the WHOLE flow.
//
// Two flops with combinational logic between them, so there is a genuine
// register-to-register path for `create_clock` + `report_checks` to constrain.
// An input-to-register design reports "No paths found" without input delays,
// and a purely combinational one has no clocked path at all -- either would
// make a smoke test pass while proving nothing about the timing arc.
module smoke_top(input clk, input a, input b, input c, output y);
  reg p, q;
  always @(posedge clk) begin
    p <= (a & b) | ~c;
    q <= ~p;
  end
  assign y = q;
endmodule
