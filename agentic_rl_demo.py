
import numpy as np
import random
from collections import defaultdict, deque

class GridWorld:
    def __init__(self, width=6, height=6, start=(0,0), goal=(5,5), walls=None, wind_prob=0.10):
        self.W = width; self.H = height
        self.start = start; self.goal = goal
        self.walls = set(walls or [])
        self.wind_prob = wind_prob
        self.reset()

    def reset(self):
        self.s = self.start
        return self.s

    def in_bounds(self, s):
        x,y = s
        return 0 <= x < self.W and 0 <= y < self.H

    def is_free(self, s):
        return self.in_bounds(s) and (s not in self.walls)

    def step(self, a):
        if random.random() < self.wind_prob:
            a = random.randint(0,3)
        x,y = self.s
        moves = [(x,y-1),(x+1,y),(x,y+1),(x-1,y)]
        ns = moves[a]
        r = -1.0; done = False
        if not self.in_bounds(ns) or ns in self.walls:
            ns = self.s; r = -2.0
        if ns == self.goal:
            r = 20.0; done = True
        self.s = ns
        return ns, r, done, {}

def bfs_shortest_path(env, start=None, goal=None):
    start = start or env.start; goal = goal or env.goal
    q = deque([start]); came = {start: None}
    dirs = [(0,-1),(1,0),(0,1),(-1,0)]
    while q:
        cur = q.popleft()
        if cur == goal: break
        for dx,dy in dirs:
            nxt = (cur[0]+dx, cur[1]+dy)
            if nxt not in came and env.is_free(nxt):
                came[nxt] = cur; q.append(nxt)
    if goal not in came: return []
    path = []; node = goal
    while node is not None:
        path.append(node); node = came[node]
    return list(reversed(path))

class AgenticDynaQ:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.40, planning_steps=25, intrinsic_loop_penalty=0.001):
        self.env=env; self.alpha=alpha; self.gamma=gamma; self.epsilon=epsilon
        self.planning_steps=planning_steps; self.intrinsic_loop_penalty=intrinsic_loop_penalty
        self.Q = defaultdict(lambda: np.zeros(4, dtype=np.float32))
        self.model_next = {}; self.model_reward = {}
        self.returns=[]; self.lengths=[]; self.best_return=-1e9; self.no_improve_streak=0
        self.visit_counts = defaultdict(int)

    def policy(self,s):
        return random.randint(0,3) if random.random()<self.epsilon else int(np.argmax(self.Q[s]))

    def update(self, s,a,r,ns,done):
        loop_pen = -self.intrinsic_loop_penalty * self.visit_counts[s]
        r_adj = r + loop_pen
        target = r_adj + (0 if done else self.gamma*np.max(self.Q[ns]))
        self.Q[s][a] += self.alpha*(target - self.Q[s][a])
        self.model_next[(s,a)] = ns; self.model_reward[(s,a)] = r_adj
        for _ in range(self.planning_steps):
            ps,pa = random.choice(list(self.model_next.keys()))
            pns = self.model_next[(ps,pa)]; pr = self.model_reward[(ps,pa)]
            ptarg = pr + self.gamma*np.max(self.Q[pns])
            self.Q[ps][pa] += self.alpha*(ptarg - self.Q[ps][pa])

    def reflect_and_adjust(self, ep_return, ep_len):
        self.returns.append(ep_return); self.lengths.append(ep_len)
        if ep_return > self.best_return:
            self.best_return = ep_return; self.no_improve_streak = 0
        else:
            self.no_improve_streak += 1
        if self.no_improve_streak > 8:
            self.epsilon = max(0.05, self.epsilon*0.9)
            self.planning_steps = min(80, int(self.planning_steps*1.2))

    def act_with_tool_if_stuck(self, s, steps):
        if steps < 30: return
        path = bfs_shortest_path(self.env, start=s, goal=self.env.goal)
        if len(path)<=1: return
        dirs = [(0,-1),(1,0),(0,1),(-1,0)]
        for i in range(len(path)-1):
            cur,nxt = path[i], path[i+1]
            dx,dy = nxt[0]-cur[0], nxt[1]-cur[1]
            if (dx,dy) in dirs:
                a = dirs.index((dx,dy))
                self.Q[cur][a] = max(self.Q[cur][a], 1.0)

def train(seed=42, episodes=280):
    random.seed(seed); np.random.seed(seed)
    walls={(2,1),(2,2),(2,3),(3,3),(4,3)}
    env = GridWorld(6,6,(0,0),(5,5),walls,wind_prob=0.10)
    agent = AgenticDynaQ(env)
    ep_returns=[]; ep_lengths=[]
    for ep in range(episodes):
        s=env.reset(); total=0.0; steps=0
        while True:
            agent.visit_counts[s]+=1
            a=agent.policy(s)
            ns,r,done,_=env.step(a)
            total+=r; steps+=1
            agent.update(s,a,r,ns,done)
            if steps%40==0 and not done:
                agent.act_with_tool_if_stuck(ns, steps)
            s=ns
            if done or steps>400: break
        agent.reflect_and_adjust(total, steps)
        ep_returns.append(total); ep_lengths.append(steps)
    return env, agent, ep_returns, ep_lengths

if __name__ == "__main__":
    env, agent, rets, lens = train()
    print("Training complete. Episodes:", len(rets))
