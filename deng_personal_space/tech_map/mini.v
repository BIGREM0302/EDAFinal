// mini.v ── 最簡單的 1-bit D 觸發器
module mini (input clk, input d, output reg q);
    always @(posedge clk)
        q <= d;
endmodule
