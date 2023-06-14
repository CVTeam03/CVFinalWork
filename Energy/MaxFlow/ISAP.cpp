#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
const int N = 1e4 + 5, M = 2e5 + 5, INF = 1e8;
int h[N], e[M], ne[M], w[M], idx, d[N], q[N], n, m, S, T, cur[N], gap[N];
long long maxflow;
void add(int a, int b, int c)
{
    e[idx] = b, ne[idx] = h[a], w[idx] = c, h[a] = idx ++;
    e[idx] = a, ne[idx] = h[b], w[idx] = 0, h[b] = idx ++;
}

void bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = T, d[T] = 0, cur[T] = h[T], gap[0] = 1;
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && !w[i])
            {
                cur[ver] = h[ver];
                d[ver] = d[t] + 1;
                gap[d[ver]] ++;
                q[++ tt] = ver;
            }
        }
    }
}
int min(int a, long long b)
{
    return a < b ? a : b;
}
long long find(int u, long long limit)
{
    if (u == T)
    {
        maxflow += limit;
        return limit;
    }
    int flow = 0;

    for (int i = cur[u]; ~i; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] + 1 == d[u] && w[i])
        {
            int t = find (ver, min(w[i], limit - flow));
            w[i] -= t, w[i ^ 1] += t, flow += t;
            if (flow == limit) return flow;
        }
    }
    -- gap[d[u]];
    if (gap[d[u]] == 0) d[S] = n + 1;
    d[u] ++;
    gap[d[u]] ++;
    return flow;
}

long long ISAP()
{
    bfs();
    while(d[S] < n)
    {
        memcpy(cur, h, sizeof h);
        find(S, INF);
    }
    return maxflow;
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

    printf("%lld\n", ISAP());

    return 0;
}