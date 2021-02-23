import numpy as np


class Agent:

    def __init__(self, n_machines=100):
        self.t = 0  # current time step
        self.n_machines = n_machines
        self.reward = np.zeros(n_machines)

    def pick_machine(self):
        # returns index of machine
        if self.t > 5:
            ind = self.reward.argmax()
        elif self.t % 5 ==0:
            ind = np.random.randint(0, self.n_machines)
        else:
            ind = np.random.randint(0, self.n_machines)
        return ind
        # return np.random.randint(0, self.n_machines)

    def get_reward(self, reward, machine_index):
        self.t += 1
        self.reward[machine_index] += reward


class Environment:

    def __init__(self, n_machines=100):
        self.n_machines = n_machines
        self.means = means = np.random.normal(1, 1, size=n_machines)

    def _interact(self, machine_index):
        assert 0 <= machine_index < self.n_machines, 'Bad machine index'
        reward = np.random.normal(self.means[machine_index], 1)
        return reward

    def run(self, time_steps=1000, n=5, verbose=False):
        total_reward = 0
        agent = Agent(n)
        self.n_machines = agent.n_machines #########################
        self.means = np.random.normal(1, 1, size=self.n_machines) ############################3
        for _ in range(time_steps):
            machine_index = agent.pick_machine()
            reward = self._interact(machine_index)
            agent.get_reward(reward, machine_index)
            total_reward += reward

        if verbose:
            print(agent.reward)
        return total_reward


np.random.seed(0)
print(Environment().run(n=100, verbose=False))