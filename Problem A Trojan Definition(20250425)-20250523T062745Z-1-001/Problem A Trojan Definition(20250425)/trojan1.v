module Trojan1(
	input clk,
	input rst,
	input r1,
	output trigger
    );

	reg [3: 0] counter;
	
	always @ (posedge clk)
		begin
			if (~rst) begin
				counter = 0;
			end
			if (r1 == 1) counter = counter + 1'b1;		
		end
	assign trigger = counter[3];

endmodule