import numpy as np
import streamlit as st

# page settings
st.set_page_config(layout="wide")

max_width = 1000
padding_top = 0
padding_right = "20%"
padding_left = "10%"
padding_bottom = 0
COLOR = "#1f1f2e"
BACKGROUND_COLOR = "#d1d1e0"

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

class Agent:

    def __init__(self, n_machines=100):
        self.t = 0  # current time step
        self.n_machines = n_machines
        self.reward = np.zeros((n_machines, 2))

    def pick_machine(self):
        # returns index of machine
        np.seterr(divide='ignore', invalid='ignore')
        if self.t > self.n_machines // 10: # try 10% of all machines
            ind = (self.reward[:,0]/self.reward[:,1]).argmax()
        elif self.t % 5 == 0:
            ind = np.random.randint(0, self.n_machines)
        else:
            ind = np.random.randint(0, self.n_machines)
        return ind
        # return np.random.randint(0, self.n_machines)

    def get_reward(self, reward, machine_index):
        self.t += 1
        self.reward[machine_index, 0] += reward
        self.reward[machine_index, 1] += 1

class Environment:

    def __init__(self, n_machines=100):
        self.n_machines = n_machines
        self.means = np.random.normal(1, 1, size=n_machines)

    def _interact(self, machine_index):
        assert 0 <= machine_index < self.n_machines, 'Bad machine index'
        reward = np.random.normal(self.means[machine_index], 1)
        return reward

    def run(self, time_steps=1000, n=100, verbose=False):
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
            # print(self.means)
        return total_reward

if __name__ == '__main__':
    np.random.seed(0)
    st.write(Environment().run(verbose=False))