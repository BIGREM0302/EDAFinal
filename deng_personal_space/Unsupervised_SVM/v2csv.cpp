#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace std;

constexpr int INF = INT_MAX;

/* ---------- 資料結構 ---------- */

struct Node {
  string name;           // 節點名稱（gate or port）
  string type;           // and / nand / dff / input ...
  string output;         // 該 gate 驅動的 wire
  vector<string> inputs; // 其餘輸入 wire
};

unordered_map<string, int> type_id;           // gate type → index
unordered_map<string, Node> nodes;            // node_name → Node
unordered_map<string, string> wire2node;      // wire → node_name
unordered_map<string, vector<string>> fanout; // node_name → 下游 nodes
vector<pair<int, string>> id_name;            // for stable csv order

/* ---------- 特徵表 ---------- */

unordered_map<string, int> LGFi, FFi, FFo, Pi, Po;

/* ---------- util ---------- */

bool valid(const string &w) { return w != "1'b0" && w != "1'b1"; }

int lgfi(const string &n) {
  auto &g = nodes[n];
  if (g.inputs.empty())
    return 1;
  int s = 0;
  for (auto &w : g.inputs) {
    if (!valid(w))
      continue;
    auto up = wire2node[w];
    s += max<int>(1, nodes[up].inputs.size());
  }
  return s;
}

/* ---------- FFi：到最近 DFF(向上) ---------- */
int ffi(const string &n) {
  if (FFi.count(n))
    return FFi[n];
  if (nodes[n].type == "dff") {
    return FFi[n] = 0;
  }
  if (nodes[n].inputs.empty())
    return FFi[n] = INF;
  int best = INF;
  for (auto &w : nodes[n].inputs) {
    if (!valid(w))
      continue;
    best = min(best, ffi(wire2node[w]));
  }
  return FFi[n] = (best == INF ? INF : best + 1);
}

/* ---------- FFo：到最近 DFF(向下) ---------- */
int ffo(const string &n) {
  if (FFo.count(n))
    return FFo[n];
  if (nodes[n].type == "dff") {
    return FFo[n] = 0;
  }
  if (fanout[n].empty())
    return FFo[n] = INF;
  int best = INF;
  for (auto &dn : fanout[n])
    best = min(best, ffo(dn));
  return FFo[n] = (best == INF ? INF : best + 1);
}

/* ---------- Pi：到最近 primary-input ---------- */
int pi(const string &n) {
  if (Pi.count(n))
    return Pi[n];
  if (nodes[n].type == "input")
    return Pi[n] = 0;
  if (nodes[n].inputs.empty())
    return Pi[n] = INF;
  int best = INF;
  for (auto &w : nodes[n].inputs) {
    if (!valid(w))
      continue;
    best = min(best, pi(wire2node[w]));
  }
  return Pi[n] = (best == INF ? INF : best + 1);
}

/* ---------- Po：到最近 primary-output ---------- */
int po(const string &n) {
  if (Po.count(n))
    return Po[n];
  if (nodes[n].type == "output")
    return Po[n] = 0;
  if (fanout[n].empty())
    return Po[n] = INF;
  int best = INF;
  for (auto &dn : fanout[n])
    best = min(best, po(dn));
  return Po[n] = (best == INF ? INF : best + 1);
}

/* ---------- Verilog 解析用小工具 ---------- */
vector<string> split_args(string s) {
  // 去掉括號後，用逗號切
  s = s.substr(s.find('(') + 1); // 到 '(' 後
  if (s.back() == ')')
    s.pop_back();
  string t;
  vector<string> v;
  stringstream ss(s);
  while (getline(ss, t, ',')) {
    t.erase(remove_if(t.begin(), t.end(), ::isspace), t.end());
    v.push_back(t);
  }
  return v;
}
vector<string> expand_ports(string line) {
  // input [3:0] a,b; → a[0]~a[3], b[0]~b[3]
  smatch m;
  vector<string> out;
  regex r(R"(\[(\d+):(\d+)\])");
  if (regex_search(line, m, r)) {
    int hi = stoi(m[1]), lo = stoi(m[2]);
    string vars = line.substr(m.position() + m.length());
    vars = regex_replace(vars, regex(R"(;|\s)"), "");
    string v;
    stringstream ss(vars);
    while (getline(ss, v, ',')) {
      for (int i = lo; i <= hi; i++)
        out.push_back(v + "[" + to_string(i) + "]");
    }
  } else {
    line = regex_replace(line, regex(R"(;|\s)"), "");
    string v;
    stringstream ss(line);
    while (getline(ss, v, ','))
      out.push_back(v);
  }
  return out;
}

/* ---------- 主要流程 ---------- */
int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " design.v out.csv\n";
    return 1;
  }
  string vin = argv[1], csv_out = argv[2];

  /* type 編碼 */
  type_id = {{"input", 0}, {"output", 1}, {"and", 2}, {"or", 3},
             {"nand", 4},  {"nor", 5},    {"not", 6}, {"buf", 7},
             {"xor", 8},   {"xnor", 9},   {"dff", 10}};

  ifstream fin(vin);
  if (!fin) {
    cerr << "Cannot open " << vin << "\n";
    return 1;
  }

  nodes.clear();
  wire2node.clear();
  fanout.clear();
  id_name.clear();
  string line;
  int nid = 0;

  while (getline(fin, line)) {
    stringstream ss(line);
    string key;
    ss >> key;
    if (key == "and" || key == "or" || key == "nand" || key == "nor" ||
        key == "xor" || key == "xnor" || key == "buf" || key == "not") {
      string rest;
      getline(ss, rest);
      size_t lp = rest.find('(');
      string name = rest.substr(0, lp);
      name.erase(remove_if(name.begin(), name.end(), ::isspace), name.end());
      auto args = split_args(rest);
      Node g{name, key, valid(args[0]) ? args[0] : "",
             vector<string>(args.begin() + 1, args.end())};
      nodes[name] = g;
      if (valid(g.output))
        wire2node[g.output] = name;
      id_name.push_back({nid++, name});
    } else if (key == "dff") {
      string rest;
      getline(ss, rest);
      size_t lp = rest.find('(');
      string name = rest.substr(0, lp);
      name.erase(remove_if(name.begin(), name.end(), ::isspace), name.end());
      auto conn = split_args(rest); // RN,SN,CK,D,Q
      Node d{name, key, valid(conn[4]) ? conn[4] : "",
             vector<string>(conn.begin(), conn.end() - 1)};
      nodes[name] = d;
      if (valid(d.output))
        wire2node[d.output] = name;
      id_name.push_back({nid++, name});
    } else if (key == "input" || key == "output") {
      string rest;
      getline(ss, rest);
      auto ports = expand_ports(rest);
      for (auto &p : ports) {
        Node n{p, key, p, {}};
        nodes[p] = n;
        wire2node[p] = p;
        id_name.push_back({nid++, p});
      }
    }
  }
  /* 建 fan-out */
  for (auto &[n, g] : nodes) {
    for (auto &w : g.inputs) {
      if (!valid(w))
        continue;
      fanout[wire2node[w]].push_back(n);
    }
  }

  /* 計算特徵 */
  for (auto &[_, n] : id_name) {
    LGFi[n] = lgfi(n);
    FFi[n] = ffi(n);
    FFo[n] = ffo(n);
    Pi[n] = pi(n);
    Po[n] = po(n);
  }

  /* 輸出 CSV */
  ofstream fout(csv_out);
  fout << "id,name,LGFi,FFi,FFo,Pi,Po,Trojan gate\n";
  for (auto &[id, n] : id_name) {
    fout << id << ',' << n << ',' << LGFi[n] << ',' << FFi[n] << ',' << FFo[n]
         << ',' << Pi[n] << ',' << Po[n] << ",0\n";
  }
  return 0;
}
