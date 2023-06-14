#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
const int N = 10010, M = 200010, INF = 1e8;
int h[N], ne[M], e[M], idx, w[M], n, m, S, T;
int d[N], cur[N], q[N];
void add(int a, int b, int c)
{
    e[idx] = b, ne[idx] = h[a], w[idx] = c, h[a] = idx ++;
    e[idx] = a, ne[idx] = h[b], w[idx] = 0, h[b] = idx ++;
}

bool bfs()
{
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    int hh = 0, tt = 0;
    while (hh <= tt)
    {
        int t = q[hh ++];
        for (int i = h[t]; i != -1; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && w[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[++ tt] = ver;
            }
        }
    }
    return false;
}
long long min(int a, long long b)
{
    return a < b ? a : b;
}
int find(int u, long long limit)
{
    if (u == T) return limit;
    long long flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && w[i])
        {
            int t = find(ver, min(w[i], limit - flow));
            if (!t) d[ver] = -1;
            w[i] -= t, w[i^ 1] += t, flow += t;
        }
    }
    return flow;
}

long long dinic()
{
    long long res = 0, flow = 0;
    while (bfs()) while (flow = find(S, INF)) res += flow;
    return res;
}

int main()
{
    scanf("%d%d%d%d", &n, &m, &S, &T);
    memset(h, -1, sizeof h);
    while (m -- ) 
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }
    
    printf("%lld\n", dinic());
    
    return 0;
}