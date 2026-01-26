import heapq
import math

class DStar:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = set()
        self.goal = None

    def set_goal(self, goal):
        self.goal = goal

    def add_obstacle(self, cell):
        self.obstacles.add(cell)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, start):
        pq = []
        heapq.heappush(pq, (0, start))
        came_from = {}
        cost = {start: 0}

        while pq:
            _, current = heapq.heappop(pq)
            if current == self.goal:
                break

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nxt = (current[0]+dx, current[1]+dy)
                if nxt in self.obstacles:
                    continue
                if nxt[0]<0 or nxt[1]<0 or nxt[0]>=self.width or nxt[1]>=self.height:
                    continue

                new_cost = cost[current] + 1
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    priority = new_cost + self.heuristic(nxt, self.goal)
                    heapq.heappush(pq, (priority, nxt))
                    came_from[nxt] = current

        return came_from
