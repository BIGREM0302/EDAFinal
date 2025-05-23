module Trojan0 (
    input wire clk,
    input wire rst,
    input wire [127:0] key,
    output reg [63:0] load
);
	wire [19:0] counter;
	lfsr_counter lfsr (rst, clk, counter);

	always @(posedge clk) begin
		integer i;
		for (i = 0; i < 64; i = i + 1) begin
			load[i] <= key[i / 8] ^ counter[i / 8];
		end
	end

endmodule


module lfsr_counter (
	input rst, clk, 
	output [19:0] lfsr
);

	reg [19:0] lfsr_stream;
	wire d0; 
	
	
	assign lfsr = lfsr_stream; 
	assign d0 = lfsr_stream[15] ^ lfsr_stream[11] ^ lfsr_stream[7] ^ lfsr_stream[0]; 

	always @(posedge clk)
		if (rst == 1) begin
			lfsr_stream <= "10011001100110011001";
		end else begin
			lfsr_stream <= {d0,lfsr_stream[19:1]}; 
		end
		
endmodule