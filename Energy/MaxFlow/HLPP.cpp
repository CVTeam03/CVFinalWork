/*
 * https://oi-wiki.org/graph/flow/max-flow/#hlpp-%E7%AE%97%E6%B3%95
*/
#include <iostream>
#include <algorithm>
#include <cstring>
#include <stack>
#include <queue>
using namespace std;
const int N = 1e4 + 5, M = 2e5 + 5, INF = 0x3f3f3f3f;
int h[N], e[M], ne[M], w[M], idx, n, m, S, T;
void add(int a, int b, int c)
{
    e[idx] = b, ne[idx] = h[a], w[idx] = c, h[a] = idx ++;
    e[idx] = a, ne[idx] = h[b], w[idx] = 0, h[b] = idx ++;
}

int ht[N + 1], gap[N];
long long ex[N + 1];  // 高度; 超额流; gap 优化 gap[i] 为高度为 i 的节点的数量
stack<int> B[N];  // 桶 B[i] 中记录所有 ht[v]==i 的v
int level = 0;    // 溢出节点的最高高度
long long min(int a, long long b)
{
    return a < b ? a : b;
}
int push(int u) 
{      // 尽可能通过能够推送的边推送超额流
  bool init = u == S;  // 是否在初始化
  for (int i = h[u]; ~i; i = ne[i]) {
    const int &v = e[i], &ww = w[i];
    if (!ww || init == false && ht[u] != ht[v] + 1)  // 初始化时不考虑高度差为1
      continue;
    int k = init ? ww : min(ww, ex[u]);
    // 取到剩余容量和超额流的最小值，初始化时可以使源的溢出量为负数。
    if (v != S && v != T && !ex[v]) B[ht[v]].push(v), level = max(level, ht[v]);
    ex[u] -= k, ex[v] += k, w[i] -= k, w[i ^ 1] += k;  // push
    if (!ex[u]) return 0;  // 如果已经推送完就返回
  }
  return 1;
}

void relabel(int u) 
{  // 重贴标签（高度）
  ht[u] = INF;
  for (int i = h[u]; ~i; i = ne[i])
    if (w[i]) ht[u] = min(ht[u], ht[e[i]]);
  if (++ht[u] < n) {  // 只处理高度小于 n 的节点
    B[ht[u]].push(u);
    level = max(level, ht[u]);
    ++gap[ht[u]];  // 新的高度，更新 gap
  }
}

bool bfs_init() 
{
  memset(ht, 0x3f, sizeof(ht));
  queue<int> q;
  q.push(T), ht[T] = 0;
  while (q.size()) {  // 反向 BFS, 遇到没有访问过的结点就入队
    int u = q.front();
    q.pop();
    for (int i = h[u]; ~i; i = ne[i]) {
      const int &v = e[i];
      if (w[i ^ 1] && ht[v] > ht[u] + 1) ht[v] = ht[u] + 1, q.push(v);
    }
  }
  return ht[S] != INF;  // 如果图不连通，返回 0
}

// 选出当前高度最大的节点之一, 如果已经没有溢出节点返回 0
int select() 
{
  while (B[level].size() == 0 && level > -1) level--;
  return level == -1 ? 0 : B[level].top();
}

long long hlpp() 
{                  // 返回最大流
  if (!bfs_init()) return 0;  // 图不连通
  for (int i = 1; i <= n; i++)
    if (ht[i] != INF) gap[ht[i]]++;  // 初始化 gap
  ht[S] = n;
  push(S);  // 初始化预流
  int u;
  while ((u = select())) {
    B[level].pop();
    if (push(u)) {  // 仍然溢出
      if (!--gap[ht[u]])
        for (int i = 1; i <= n; i++)
          if (i != S && i != T && ht[i] > ht[u] && ht[i] < n + 1)
            ht[i] = n + 1;  // 这里重贴成 n+1 的节点都不是溢出节点
      relabel(u);
    }
  }
  return ex[T];
}

int main() {
  scanf("%d%d%d%d", &n, &m, &S, &T);
  memset(h, -1, sizeof h);
  for (int i = 1, u, v, w; i <= m; i++) {
    scanf("%d%d%d", &u, &v, &w);
    add(u, v, w);
  }
  printf("%lld", hlpp());
  return 0;
}