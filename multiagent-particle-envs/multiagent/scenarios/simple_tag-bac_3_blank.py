import numpy as np
from multiagent.core import World, Agent, Landmark, wall
from multiagent.scenario import BaseScenario
import random

random.seed()

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 2
        num_adversaries = 2
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2

        num_walls=4

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' %i
            agent.name_id=i
            agent.collide = True
            agent.silent = True
            agent.shooting_angle=0
            agent.test=[0,0]
            agent.bonus=1
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            #agent.accel = 3.0 if agent.adversary else 4.0
            agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 2.0 if agent.adversary else 2.0
        # # add landmarks
        # world.landmarks = [Landmark() for i in range(num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d' % i
        #     landmark.collide = True
        #     landmark.movable = False
        #     landmark.size = 0.2
        #     landmark.boundary = False

        # make initial conditions
        world.obstacles = [wall() for i in range(num_walls)]
        # world.obstacles[0].xy=[[2200,0],[2500,800]]
        # world.obstacles[1].xy=[[3700,1200],[4000,2000]]
        # world.obstacles[2].xy=[[1500,1800],[2700,2100]]
        # world.obstacles[3].xy=[[0,3100],[2000,3400]]
        # for i in range(4):
        #     t=world.obstacles[i].xy
        #     world.obstacles[i+4].xy=[[5000-t[1][0],8000-t[1][1]],
        #                             [5000-t[0][0],8000-t[0][1]]]
        world.obstacles[0].xy=[[-300,-300],[5300,0]]
        world.obstacles[1].xy=[[-300,-300],[0,8300]]
        world.obstacles[2].xy=[[5000,-300],[5300,8300]]
        world.obstacles[3].xy=[[-300,8000],[5300,8300]]
        
        for i in range(num_walls):
            world.obstacles[i].xy=np.array(world.obstacles[i].xy)/100*0.075



        self.reset_world(world)
        return world


    def reset_world(self, world):
        
        def check_in_wall(pos):
            for _,wall in enumerate(world.obstacles):
                if (pos[0]>wall.xy[0][0]) and (pos[0]<wall.xy[1][0]) and (pos[1]>wall.xy[0][1]) and (pos[1]<wall.xy[1][1]):
                    return(True)
            return(False)
        initial_pos=[[0,0],[0,0],[0,0],[0,0]]
        # initial_pos[0]=[random.randint(10,4990),random.randint(10,7990)]
        # initial_pos[1]=[random.randint(10,4990),random.randint(10,7990)]
        # initial_pos[2]=[random.randint(10,4990),random.randint(10,7990)]
        #initial_pos[3]=[random.randint(10,4990),random.randint(10,7990)]
        initial_pos[0]=[600,600]
        initial_pos[1]=[600,800]
        initial_pos[2]=[5000-600,8000-600]
        initial_pos[3]=[5000-600,8000-800]

        t=random.randint(0,10)
        t=3
        if t>2:
            for i in range(4):
                initial_pos[i]=[random.randint(10,4990),random.randint(10,7990)]
                while check_in_wall(initial_pos[i]):
                    initial_pos[i]=[random.randint(10,4990),random.randint(10,7990)]

        # initial_pos[3]=[5000-600,8000-800]
        #initial_pos[2]=[5000-200,600]
        initial_pos=np.array(initial_pos)/100*0.075

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for i,agent in enumerate(world.agents):
            agent.state.p_pos = initial_pos[i]
            agent.state.shooting_angle=random.randint(0,8)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        agent_reward, adv_rew = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return agent_reward, adv_rew

    def check_hit_wall(self,wall,line1,p):
        v=[[0,0] for i in range(4)]
        v[0]=[wall.xy[0][0],wall.xy[0][1]]
        v[1]=[wall.xy[0][0],wall.xy[1][1]]
        v[2]=[wall.xy[1][0],wall.xy[1][1]]
        v[3]=[wall.xy[1][0],wall.xy[0][1]]
        for i in range(4):
            line2=self.calculate_line(v[i],v[(i+1)%4])
            det=line1[0]*line2[1]-line2[0]*line1[1]
            if det==0:
                return False
            else:
                x=(line2[1]*line1[2]-line1[1]*line2[2])/det
                y=(line1[0]*line2[2]-line2[0]*line1[2])/det
                if (x>=min(v[i][0],v[(i+1)%4][0])) and (x<=max(v[i][0],v[(i+1)%4][0])):
                    if (y>=min(v[i][1],v[(i+1)%4][1])) and (y<=max(v[i][1],v[(i+1)%4][1])):
                        if (x>=min(p[0][0],p[1][0])) and (x<=max(p[0][0],p[1][0])):
                            if (y>=min(p[1][1],p[0][1])) and (y<=max(p[0][1],p[1][1])):
                                return True

    def check_hit_wall2(self,wall,p1,p2):
        v=np.array([
        [wall.xy[0][0],wall.xy[0][1]],
        [wall.xy[0][0],wall.xy[1][1]],
        [wall.xy[1][0],wall.xy[1][1]],
        [wall.xy[1][0],wall.xy[0][1]]])
        # print("wall",wall.xy)
        # print(v)
        # print("pppppp",p1,p2)
        for i in range(4):
            t=self.get_intersection(v[i],v[(i+1)%4],p1,p2)
            if t is not None:
                #print(v[i],v[(i+1)%4])
                return t
        return None

    def check_intersection(self,line1,line2,p,p1):
        det=line1[0]*line2[1]-line2[0]*line1[1]
        if det==0:
            return False
        else:
            x=(line2[1]*line1[2]-line1[1]*line2[2])/det
            y=(line1[0]*line2[2]-line2[0]*line1[2])/det
            if (x>=min(p1[0][0],p1[1][0])) and (x<=max(p1[0][0],p1[1][0])):
                if (y>=min(p1[0][1],p1[1][1])) and (y<=max(p1[0][1],p1[1][1])):
                    if (x>=min(p[0][0],p[1][0])) and (x<=max(p[0][0],p[1][0])):
                        if (y>=min(p[1][1],p[0][1])) and (y<=max(p[0][1],p[1][1])):
                            return True
    
    def calculate_line(self,x1,x2):
        return [x2[1]-x1[1],x1[0]-x2[0],(x2[1]-x1[1])*x1[0]+(x1[0]-x2[0])*x1[1]]
    
    def get_intersection(self,a1, a2, b1, b2) :
        """
        :param a1: (x1,y1) line segment 1 - starting position
        :param a2: (x1',y1') line segment 1 - ending position
        :param b1: (x2,y2) line segment 2 - starting position
        :param b2: (x2',y2') line segment 2 - ending position
        :return: point of intersection, if intersect; None, if do not intersect
        #adopted from https://github.com/LinguList/TreBor/blob/master/polygon.py
        """

        def perp(a) :
            b = np.empty_like(a)
            b[0] = -a[1]
            b[1] = a[0]
            return b

        da = a2-a1
        db = b2-b1
        dp = a1-b1
        dap = perp(da)
        denom = np.dot( dap, db)
        num = np.dot( dap, dp )
        intersct = np.array((num/(denom.astype(float)+0.00000001))*db + b1) #TODO: check divide by zero!

        delta = 1e-3
        condx_a = min(a1[0], a2[0])-delta <= intersct[0] and max(a1[0], a2[0])+delta >= intersct[0] #within line segment a1_x-a2_x
        condx_b = min(b1[0], b2[0])-delta <= intersct[0] and max(b1[0], b2[0])+delta >= intersct[0] #within line segment b1_x-b2_x
        condy_a = min(a1[1], a2[1])-delta <= intersct[1] and max(a1[1], a2[1])+delta >= intersct[1] #within line segment a1_y-b1_y
        condy_b = min(b1[1], b2[1])-delta <= intersct[1] and max(b1[1], b2[1])+delta >= intersct[1] #within line segment a2_y-b2_y
        if not (condx_a and condy_a and condx_b and condy_b):
            intersct = None #line segments do not intercept i.e. interception is away from from the line segments

        return intersct

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew -= 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 50
    
        
        p=np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])

        bonus=np.array([[2500-250,4000-250],[2500+250,4000+250]])/100*0.075

        pp=np.array(agent.state.p_pos) #1

        if (pp[0]>bonus[0][0]) and (pp[0]<bonus[1][0]) and (pp[1]>bonus[0][1]) and (pp[1]<bonus[1][1]):
            agent.bonus=2
        a1=np.array(p[agent.shooting_angle]*0.01+pp)
        a2=np.array(p[(agent.shooting_angle+1) % 8]*0.01+pp)
        
        # rew_opponent=0
        # for i, a in enumerate(adversaries):
        #     pos=a.state.p_pos #2
        #     # if not((((pp[1]-pos[1])*(a1[0]-pp[0])+(pos[0]-pp[0])*(a1[1]-pp[1]))*
        #     #     ((pp[1]-pos[1])*(a2[0]-pp[0])+(pos[0]-pp[0])*(a2[1]-pp[1])))<0):
        #     #     continue
        #     l=self.calculate_line(pp,pos)
        #     l2=self.calculate_line(a1,a2)
        #     if self.check_intersection(l,l2,[pp,pos],[a1,a2]):
        #         hit=True
        #         for p in world.obstacles:
        #             if self.check_hit_wall(p,l,[pp,pos]):
        #                 hit=False
        #                 break
        #         if hit:
        #             rew+=50
        #             rew_opponent-=50


        rew_opponent=[0 for i in range(4)]

        agent.test=[0,0]
        for i, a in enumerate(adversaries):
            pos=np.array(a.state.p_pos) #2
            if np.sqrt(np.sum(np.square(pos - pp)))>2:
                 continue
            if self.get_intersection(pp,pos,a1,a2) is not None:
                hit=True
                for _,p in enumerate(world.obstacles):
                    t=self.check_hit_wall2(p,pp,pos)
                    if t is not None:
                        agent.test=t
                        hit=False
                        break
                if hit:
                    rew+=50*agent.bonus
                    rew_opponent[a.name_id]-=50*agent.bonus


            


        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # def bound(x):
        #     if x < 0.9:
        #         return 0
        #     if x < 1.0:
        #         return (x - 0.9) * 10
        #     return min(np.exp(2 * x - 2), 10)
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        return rew,rew_opponent

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in agents:
                rew -= 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for ag in agents:
                if self.is_collision(ag, agent):
                    rew -= 50

        
        q=np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
        pp=np.array(agent.state.p_pos) #1

        bonus=np.array([[2500-250,4000-250],[2500+250,4000+250]])/100*0.075
        if (pp[0]>bonus[0][0]) and (pp[0]<bonus[1][0]) and (pp[1]>bonus[0][1]) and (pp[1]<bonus[1][1]):
            agent.bonus=2

        a1=np.array(q[agent.shooting_angle]*0.01+pp)
        a2=np.array(q[(agent.shooting_angle+1) % 8]*0.01+pp)
        
        # for a in agents:
        #     pos=a.state.p_pos #2
        #     # if not((((pp[1]-pos[1])*(a1[0]-pp[0])+(pos[0]-pp[0])*(a1[1]-pp[1]))*
        #     #     ((pp[1]-pos[1])*(a2[0]-pp[0])+(pos[0]-pp[0])*(a2[1]-pp[1])))<0):
        #     #     continue
        #     l=self.calculate_line(pp,pos)
        #     l2=self.calculate_line(a1,a2)
        #     if self.check_intersection(l,l2,[pp,pos],[a1,a2]):
        #         hit=True
        #         for p in world.obstacles:
        #             if self.check_hit_wall(p,l,[pp,pos]):
        #                 hit=False
        #                 break
        #         if hit:
        #             rew+=50
        #             rew_opponent-=50
        agent.test=[0,0]
        rew_opponent=[0 for i in range(4)]
        for a in agents:
            pos=np.array(a.state.p_pos) #2
            if np.sqrt(np.sum(np.square(pos - pp)))>2:
                 continue
            if self.get_intersection(pp,pos,a1,a2)is not None:
                hit=True
                for _,p in enumerate(world.obstacles):
                    t=self.check_hit_wall2(p,pp,pos)
                    if t is not None:
                        agent.test=t
                        hit=False
                        break
                if hit:
                    rew+=50*agent.bonus
                    rew_opponent[a.name_id]-=50*agent.bonus

        return rew,rew_opponent

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)
        #     if not other.adversary:
        #         other_vel.append(other.state.p_vel)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
            other_pos.append((other.state.p_pos - agent.state.p_pos)/6)
            
            other_vel.append(other.state.p_vel- agent.state.p_vel)
       
        tttt=np.concatenate([agent.state.p_vel] + [agent.state.p_pos/6] + other_pos + other_vel)
        tt=np.array([0,0,0,0,0,0,0,0])
        tt[agent.shooting_angle]=1
        return np.concatenate((tttt,tt))
