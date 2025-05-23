module Trojan8(
   input  wire [7:0] a, b, c, d, e,
   input  wire [2:0] sel,
   output wire [15:0] y
);
   wire [15:0] t1, t2, t3, t4, t5, t6, t7;


   assign t1 = a * (b + c);
   assign t2 = (a * b) + (a * c);
   assign t3 = (d + e) * (a + b); 
   assign t4 = (d * a) + (d * b) + (e * a) + (e * b);
   assign t5 = (t1 + t4) ^ (t3 & 16'h00FF);
   assign t6 = ((t2 * 2) + t5) ^ (t3 >> 1); 
   assign t7 = (t6 + (t1 ^ t2)) * ((a + c) & 8'h0F);
   assign y = (sel == 3'b000) ? t1 :
              (sel == 3'b001) ? t2 :
              (sel == 3'b010) ? t3 :
              (sel == 3'b011) ? t4 :
              (sel == 3'b100) ? t5 :
              (sel == 3'b101) ? t6 :
              (sel == 3'b110) ? t7 :
              ((t1 + t2 + t3) ^ (t4 & 16'hF0F0));
endmodule