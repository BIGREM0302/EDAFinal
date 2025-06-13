# Build Environment
```
conda env create -f environment.yml
```
# Verilog to DFG
```bash
# 最簡用法：同目錄放好 generate_dfg.py
python run_pipeline.py trojan0.v
```
就會產出<filename>_dfg.txt
