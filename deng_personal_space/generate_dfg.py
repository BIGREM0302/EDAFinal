#!/usr/bin/env python3
import sys

from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_dfg.py <verilog_file> [TopModule]")
        sys.exit(1)

    verilog_file = sys.argv[1]
    top_module = sys.argv[2] if len(sys.argv) > 2 else "Trojan0"

    # 1. 建立 Dataflow 分析器
    analyzer = VerilogDataflowAnalyzer([verilog_file], top_module)
    analyzer.generate()

    # 2. 拿到 binddict 並呼叫 tostr()
    binddict = analyzer.getBinddict()
    for signal, binds in sorted(binddict.items(), key=lambda x: str(x[0])):
        for bvi in binds:
            # bvi.tostr() 直接印出 "LHS = RHS" 形式的 Dataflow
            print(bvi.tostr())


if __name__ == "__main__":
    main()
