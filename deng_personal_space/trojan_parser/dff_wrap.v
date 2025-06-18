// 把 simplemap 產生的新式 $dff → dff
module \$dff (input C, D, output Q);
    dff _TECHMAP_REPLACE_ (.clk(C), .d(D), .q(Q));
endmodule

// 若你的 simplemap 產生舊式 \$_DFF_P__，也順手包一下
module \$_DFF_P_ (input C, D, output Q);
    dff _TECHMAP_REPLACE_ (.clk(C), .d(D), .q(Q));
endmodule
