module(n0, n1, n2, n3, n4, n8);

input [1:0] n0, n1;
input n2, n4;
input [2:0] n8;
output [1:0] n3;
output [1:0] n9;

wire n5, n6, n7;

and g1(
    .A(n0[1]), 
    .B(n1[0]), 
    .Y(n5)
);
not g2(
    .A(n5),
    .Y(n6)
);
xor g3(
    .A(n2),
    .B(n6),
    .Y(n3[1])
);
or g4(
    .A(n0[0]),
    .B(n1[1]),
    .Y(n7)
);
dff g5(
    .CK(n4),
    .D(n7),
    .Q(n3[0]),
    .RN(1'b1),
    .SN(1'b1)
);

assign n4 = n8[0];
assign n9[1:0] = {n8[1], n8[2]};

endmodule