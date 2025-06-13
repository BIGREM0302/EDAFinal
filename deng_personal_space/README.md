## `run_pipeline.py` 整體架構一覽

### 1. 專案檔案結構

```text
project_root/
├─ flatten.ys              # 你的 Yosys 範本（保留讀檔、hierarchy 等指令即可）
├─ generate_dfg.py         # 既有的 PyVerilog DFG 產生器
├─ run_pipeline.py         # ★ 本次自動化腳本（v1.2）
└─ <your verilog>.v        # 待分析的原始 SystemVerilog 檔
```

### 2. 依賴套件與工具

| 類別             | 名稱                                                                          | 建議安裝方式                                                             |
| -------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Python**     | `pyverilog`（≥ 1.3.0 develop 分支）                                             | `pip install "git+https://github.com/PyHDI/Pyverilog.git@develop"` |
|                | `ply`（隨 PyVerilog 依賴）                                                       | 自動安裝                                                               |
| **EDA 工具**     | **Yosys** ≥ 0.12                                                            | 官方 release 或 Homebrew (`brew install yosys`)                       |
| **Python 標準庫** | `argparse`, `pathlib`, `subprocess`, `tempfile`, `re`, `shutil`, `textwrap` | 內建                                                                 |

> **建議環境**：Python 3.9+（與 PyVerilog 開發分支相容），Linux/macOS 皆可

### 3. 腳本工作流程

1. **輸入校驗**

   * 確認 `<filename>.v` 存在；否則直接停止。
2. **偵測頂層 Module**

   * 快速掃描前 N 行，解析第一個 `module <name>` 作為 *top*。
3. **動態產生暫存 Yosys 腳本**

   * 以 `flatten.ys` 為範本 → 注入下列三行

     ```yosys
     read_verilog -sv <filename>.v
     hierarchy -check -top <top>
     write_verilog -noattr flattened_<filename>.v
     ```
   * 其餘你的 pass 指令（如 `proc`、`flatten` 等）保持不動。
4. **呼叫 Yosys**

   * `subprocess.run(["yosys", "-s", tmp_ys])`
   * stdout/stderr 皆捕捉並在失敗時列印最後 20 行。
5. **再次確認實際頂層**

   * 若 Yosys 重新命名（`\`字元），重新抓取 `flattened_*.v` 的第一個 `module` 名稱。
6. **呼叫 `generate_dfg.py`**

   * `python generate_dfg.py flattened_<filename>.v <TopModule>`
   * 產生的 DFG 內容重導向到 `<prefix>_dfg.txt`。
7. **輸出摘要 & 清理**

   * 顯示生成檔案路徑（flattened netlist 與 DFG）。
   * 刪除暫存 `.ys` 檔（保留 Yosys log 以便除錯）。

### 4. 指令用法

```bash
# 最簡用法：同目錄放好 flatten.ys / generate_dfg.py
python run_pipeline.py trojan0.v
```

| 參數                                  | 作用                         | 範例                         |
| ----------------------------------- | -------------------------- | -------------------------- |
| `-y PATH` / `--yosys-template PATH` | 指定自訂的 flatten 參考檔          | `-y ./my_flatten.ys`       |
| `-g PATH` / `--generator PATH`      | 指向不同版本的 generate\_dfg.py   | `-g tools/generate_dfg.py` |
| `-o PREFIX` / `--output PREFIX`     | 自訂輸出前綴                     | `-o results/trojan0`       |
| `-k` / `--keep-temp`                | 保留暫存 \*.ys 檔做 debug        | `--keep-temp`              |
| `-v` / `--verbose`                  | 顯示完整 Yosys / PyVerilog log | `-v`                       |

### 5. 成功執行後的檔案

```text
flattened_trojan0.v   # 屬性剝除、已扁平化的 Verilog
trojan0_dfg.txt       # Data-flow graph，格式由 generate_dfg.py 決定
yosys_trojan0.log     # (v1.2+) 完整 Yosys log，出錯時好追蹤
```

### 6. 常見錯誤排查

| 錯誤訊息                                | 原因                                | 解法                                                      |
| ----------------------------------- | --------------------------------- | ------------------------------------------------------- |
| `ParseError … before: "*"`          | PyVerilog 不支援 `(* … *)` 屬性        | 已用 `-noattr` 移除；若仍出現，確認手動加入的 `pragma` 是否主動寫死            |
| `DefinitionError: No such module …` | 傳給 generate\_dfg.py 的 top 名稱找不到   | v1.2 已重新掃描 `flattened_*.v`；若還有問題，檢查 Yosys 是否幫忙加反斜線 `\\` |
| `syntax error near <sv keyword>`    | PyVerilog 2005 parser 不認得部分 SV 語法 | 在 Yosys 階段加 `-sv` 讀檔並轉回 2005；或用 `sv2v` 先轉               |

