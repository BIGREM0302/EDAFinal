module(n0, n1, n2, n3, n4);

input [1:0] n0, n1;
input n2, n4;
output [1:0] n3;

wire n5, n6, n7;
and g1(n5, n0[1], n1[0]);
not g2(n6, n5);
xor g3(n3[1], n2, n6);
or g4(n7, n0[0], n1[1]);
dff g5(.RN(1'b1), .SN(1'b1), .CK(n4), .D(n7), .Q(n3[0]));

endmodule