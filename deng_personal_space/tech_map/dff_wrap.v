// dff_wrap.v ── 把 $dff / $_DFF_P_* 轉成名為 dff 的 cell
// ── 1) simplemap 產生的新式 $dff ------------------------------
module \$dff (input C, D, output Q);
    dff _TECHMAP_REPLACE_ (.clk(C), .d(D), .q(Q));
endmodule

// ── 2) 舊版 simplemap 可能留下的 $_DFF_P_ --------------------
module \$_DFF_P_ (input C, D, output Q);         // 無 reset
    dff _TECHMAP_REPLACE_ (.clk(C), .d(D), .q(Q));
endmodule
