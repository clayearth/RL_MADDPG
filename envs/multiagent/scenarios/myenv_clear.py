import math
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def __init__(self):
        self.hit_radius = 0.4
        self.mode = 0 # 0->"train"; 1->"evaluate for one esposide"
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.2 if agent.adversary else 0.2
            agent.max_speed = 5.0 if agent.adversary else 5.0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.25
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        # goal = np.random.choice(world.landmarks)
        goal = world.landmarks[0]
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal
        # # set random initial states
        if self.mode == 0: # for training
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            # # set random initial states
            # for agent in world.agents:
            #     agent.state.p_pos = np.random.uniform(-5, +5, world.dim_p)
            #     agent.state.p_vel = np.zeros(world.dim_p)
            #     agent.state.c = np.zeros(world.dim_c)
            #     agent.state.palstance = np.zeros(1)

            for agent in world.agents:
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.palstance = np.zeros(1)
                if agent.adversary:
                    # agent.state.p_pos = np.random.uniform(-10, +10, world.dim_p)
                    dis = 50
                    rdm = np.random.random()*2*math.pi
                    agent.state.p_pos = world.landmarks[0].state.p_pos + dis*np.array([math.sin(rdm),math.cos(rdm)])
                    agent.state.p_vel = np.zeros(world.dim_p)
                    pos_dif = agent.goal_a.state.p_pos-agent.state.p_pos
                    agent.state.p_vel[1] = math.atan2(pos_dif[1],pos_dif[0])#+math.pi/6*np.random.normal(0,1)
                    agent.state.c = np.zeros(world.dim_c)
                else:
                    dis = 1
                    rdm = np.random.random()*2*math.pi
                    agent.state.p_pos = world.landmarks[0].state.p_pos + dis*np.array([math.sin(rdm),math.cos(rdm)])

                    rdm_adversary = np.random.choice(self.adversaries(world))
                    pos_dif = rdm_adversary.state.p_pos-agent.state.p_pos
                    agent.state.p_vel[1] = math.atan2(pos_dif[1],pos_dif[0])#+math.pi/6*np.random.normal(0,1)
                    # print(agent.state.p_vel[1]*180/math.pi)

        elif self.mode == 1: # for testing
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            for agent in world.agents:
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.palstance = np.zeros(1)
                if agent.adversary:
                    dis = 50
                    rdm = np.random.random()*2*math.pi
                    agent.state.p_pos = world.landmarks[0].state.p_pos + dis*np.array([math.sin(rdm),math.cos(rdm)])
                    agent.state.p_vel = np.zeros(world.dim_p)
                    pos_dif = agent.goal_a.state.p_pos-agent.state.p_pos
                    agent.state.p_vel[1] = math.atan2(pos_dif[1],pos_dif[0])#+math.pi/6*np.random.normal(0,1)
                    agent.state.c = np.zeros(world.dim_c)
                else:
                    dis = 1
                    rdm = np.random.random()*2*math.pi
                    agent.state.p_pos = world.landmarks[0].state.p_pos + dis*np.array([math.sin(rdm),math.cos(rdm)])

                    rdm_adversary = np.random.choice(self.adversaries(world))
                    pos_dif = rdm_adversary.state.p_pos-agent.state.p_pos
                    agent.state.p_vel[1] = math.atan2(pos_dif[1],pos_dif[0])#+math.pi/6*np.random.normal(0,1)

    # def benchmark_data(self, agent, world):
    #     # returns data for benchmarking purposes
    #     if agent.adversary:
    #         return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
    #     else:
    #         # TODO 修改benchmark
    #         dists = []
    #         # for l in world.landmarks:
    #         #     dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
    #         adversary_agents = self.adversaries(world)
    #         for a in adversary_agents:
    #             dists.append(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
    #         return tuple(dists)

    # return every state of agents' dones
    def done(self, agent, world):
        if self.mode==0:
            if not agent.adversary:
                for adversary_agent in self.adversaries(world):
                    if np.sqrt(np.sum(np.square(agent.state.p_pos - adversary_agent.state.p_pos))) < self.hit_radius:
                        return True
                return False
            else:
                if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < self.hit_radius:
                    return True
                else:
                    return False
        elif self.mode==1:
            if not agent.adversary:
                for adversary_agent in self.adversaries(world):
                    if np.sqrt(np.sum(np.square(agent.state.p_pos - adversary_agent.state.p_pos))) < self.hit_radius:
                        return 1
                return 0
            else:
                if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < self.hit_radius:
                    return -1
                else:
                    return 0

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world, mode = None):
        if mode is None:
            mode = self.mode
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        agents = self.good_agents(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = 0
            hit_rew = 0
            for a in adversary_agents:
                # adv_rew += 0.5*np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) # every adversary's dis to goal
                # adv_rew -= 0.1 * np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) # every agent's dis to adversary
                mis_dis = min([np.sqrt(np.sum(np.square(a.state.p_pos - good_a.state.p_pos))) for good_a in agents])
                pos_dif = a.state.p_pos-agent.state.p_pos
                if np.sqrt(np.sum(np.square(pos_dif))) > 0:#5:
                    real_angle = math.atan2(pos_dif[1],pos_dif[0])
                    ABVR=agent.state.p_vel[1] - real_angle # angle between v_p and real_angle
                    guide_angle=np.minimum(2*math.pi-abs(ABVR),abs(ABVR))  #between [0,pi], reward increases when guide angle go to 0.
                    adv_rew += agent.state.p_vel[0] * (math.cos(guide_angle)/(math.sin(guide_angle)+0.1) - 1/(math.sqrt(3)+0.2))
                # if np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos))) <= self.hit_radius: # big reward for hit
                #         hit_rew += 5
                # else:0
                #     adv_rew -= 0.2 * mis_dis
                if mis_dis <= self.hit_radius: # big reward for hit
                    hit_rew += 20
        # if mode:
        #     print("\treward:\t" + str(adv_rew + hit_rew))
        return adv_rew + hit_rew

    def adversary_reward(self, agent, world, mode = None):
        if mode is None:
            mode = self.mode
        # TODO solve why adversary don't move
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            pos_dif = agent.goal_a.state.p_pos-agent.state.p_pos
            if np.sqrt(np.sum(np.square(pos_dif))) > 0:#5:
                real_angle = math.atan2(pos_dif[1],pos_dif[0])
                ABVR=agent.state.p_vel[1] - real_angle # angle between v_p and real_angle
                guide_angle=np.minimum(2*math.pi-abs(ABVR),abs(ABVR))  #between [0,pi], reward increases when guide angle go to 0.
                dis_reward = agent.state.p_vel[0] * (math.cos(guide_angle)/(math.sin(guide_angle)+0.1) - 1/(math.sqrt(3)+0.2))
            # else:
            #     dis_reward = - 0.2 * np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            hit_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) <= self.hit_radius: # big reward for hit
                hit_rew += 20
            # else:
            #     for a in self.good_agents(world):
            #         if np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos))) < self.hit_radius: # be hitten big bad reward
            #             hit_rew -= 10000
            #             break
            # if mode:
            #     print("\treward:\t" + str(dis_reward + hit_rew))
            return dis_reward + hit_rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            # print(entity.state.p_pos)
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_pos.append(other.state.p_vel)
        # if agent.adversary:
        #     print('adv_v',[agent.state.p_vel[0],agent.state.p_vel[1]*57.29577951308232])
        # else:
        #     print('good_v',[agent.state.p_vel[0],agent.state.p_vel[1]*57.29577951308232])
            

        # if self.mode:
        #     print(agent.name + "\t" + ("adversary" if agent.adversary else "good agent")
        #           + "\tpos: " + str(agent.state.p_pos) + "\tvel:" + str(agent.state.p_vel) + "\t"
        #           + "%.6f" %np.sqrt(np.sum(np.square(agent.state.p_vel))),end="")
        
        # print(np.concatenate([agent.state.p_vel] + [agent.state.palstance] + entity_pos + other_pos))
        if not agent.adversary: # 如果是追击者
            # return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
            return np.concatenate([agent.state.p_vel] + [agent.state.palstance] + entity_pos + other_pos)
        else: # 如果是逃逸者
            # print(np.concatenate(entity_pos + other_pos))
            return np.concatenate([agent.state.p_vel] + [agent.state.palstance] + entity_pos + other_pos)

    def info(self, agent, world):
        if self.mode:
            # get all info by per step
            infos = {}
            infos["n_adv"] = len(self.adversaries(world))
            infos["n_good"] = world.num_agents - len(self.adversaries(world))
            infos["adversary"] = [[] for i in range(3)] # 位置、速度、奖励
            infos["goodagent"] = [[] for i in range(3)]
            for a in world.agents:
                if a.adversary:
                    infos["adversary"][0].append(a.state.p_pos)
                    infos["adversary"][1].append(a.state.p_vel)
                    infos["adversary"][2].append(self.adversary_reward(a,world,mode=0))
                else:
                    infos["goodagent"][0].append(a.state.p_pos)
                    infos["goodagent"][1].append(a.state.p_vel)
                    infos["goodagent"][2].append(self.agent_reward(a,world,mode=0))
            # print(infos)
        else:
            infos = None
        return infos
