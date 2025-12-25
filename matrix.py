import sys

def matrix_chain_order(p):

    n = len(p) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]

    for l in range(2, n + 1):  # 链长度
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = sys.maxsize
            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + p[i-1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k
    return m, s

def print_optimal_parens(s, i, j):
    """打印最优括号化方案"""
    if i == j:
        print(f"A{i}", end="")
    else:
        print("(", end="")
        print_optimal_parens(s, i, s[i][j])
        print_optimal_parens(s, s[i][j] + 1, j)
        print(")", end="")

if __name__ == "__main__":
    # 示例：矩阵维度
    p = [30, 35, 15, 5, 10, 20, 25]
    m, s = matrix_chain_order(p)

    print("最小计算代价:", m[1][len(p)-1])
    print("最优括号化方案:", end=" ")
    print_optimal_parens(s, 1, len(p)-1)
    print()
