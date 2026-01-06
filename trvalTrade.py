import math
import random

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def total_length(points, tour):
    return sum(euclid(points[tour[i]], points[tour[(i+1) % len(tour)]]) for i in range(len(tour)))

def nearest_neighbor(points, start=0):
    n = len(points)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: euclid(points[current], points[j]))
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour

def two_opt(points, tour, max_iters=1000):
    n = len(tour)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n-1):
            for k in range(i+2, n):
                if i == 0 and k == n-1:  # 避免断开回路同一边
                    continue
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[k], tour[(k+1) % n]
                before = euclid(points[a], points[b]) + euclid(points[c], points[d])
                after = euclid(points[a], points[c]) + euclid(points[b], points[d])
                if after + 1e-12 < before:
                    # 反转区间 (i+1 ... k)
                    tour[i+1:k+1] = reversed(tour[i+1:k+1])
                    improved = True
    return tour

def tsp_heuristic(points, restarts=10):
    n = len(points)
    best_tour, best_cost = None, float('inf')
    starts = {0} | set(random.sample(range(n), min(restarts-1, n-1)))
    for s in starts:
        tour = nearest_neighbor(points, start=s)
        tour = two_opt(points, tour)
        cost = total_length(points, tour)
        if cost < best_cost:
            best_tour, best_cost = tour, cost
    # 回路形式返回
    return best_tour + [best_tour[0]], best_cost

if __name__ == "__main__":
    random.seed(42)
    pts = [(random.random()*100, random.random()*100) for _ in range(200)]
    route, cost = tsp_heuristic(pts, restarts=8)
    print("Heuristic route length:", cost)
    print("First 10 nodes:", route[:10])
