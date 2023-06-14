#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
const int N = 1010, M = 2e5 + 5;
int h[N], ne[M], e[M], w[M], idx, n, m, S, T;
int pre[N], d[N], q[N];
bool st[N];

void add(int a, int b, int c)
{
    e[idx] = b, ne[idx] = h[a], w[idx] = c, h[a] = idx ++;
    e[idx] = a, ne[idx] = h[b], w[idx] = 0, h[b] = idx ++;
}
bool bfs()
{
    memset(st, false, sizeof st);
    q[0] = S, d[S] = 1e8;
    int hh = 0, tt = 0;
    while (hh <= tt)
    {
        int t = q[hh ++];
        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (!st[j] && w[i])
            {
                d[j] = min(d[t], w[i]);
                st[j] = true;
                pre[j] = i;
                if (j == T) return true;
                q[++ tt] = j;
            }
        }
    }
    return false;
}

long long EK()
{
    long long res = 0;
    while (bfs())
    {
        res += d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
            w[pre[i]] -= d[T], w[pre[i] ^ 1] += d[T];
    }
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
    
    printf("%lld\n", EK());
    
    return 0;
}