from qutip import *
import numpy as np
import numpy.linalg
from math import *
from copy import copy
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from qutip import operators
from scipy.special import mathieu_a

def kronecker(i, j):
    return float(i == j)

# physical constants
# plank constant
hplank = 6.626e-34
# reduced plank constant
hbar = hplank / 2 / pi
# electron charge
e = 1.6e-19
# magnetic flux quantum
Fi0 = hplank / 2 / e
GHz = 1e9

cattet = lambda x: x + 1 - (x + 1) % 2
Ek_func = lambda x, Ec, Ej: mathieu_a(cattet(x), -2 * Ej / Ec) * Ec / 4

class Qubit(object):
    def __init__(self, Cq=1.01e-13, Cx=1e-16, Cg=1e-14, Ic=30e-9):
        self.Cx = Cx
        self.Cq = Cq
        self.Cg = Cg
        self.C = self.Cq + self.Cx + self.Cg
        self.Ic = 30e-9


def dummyExternalDrive2D(t, epsilon, f_c):
    theta = pi / 2 - (2 * pi * epsilon + f_c) * t
    return [-sin(theta) , cos(theta)]


class TransmonEnv(gym.Env):
    def __init__(self, qubit=Qubit(), 
                 nmax=5, Q=1e3, T1=1e9,
                 Zr=50, extFlux=0., 
                 extVoltage=0., omega=6, 
                 max_stamp=300,
                 RWA=True, temp=0.0024, 
                 fock_space_size=3, 
                 use_exact_energy_levels=True,
                 f_c=12e9, nn=lambda x, args: 0., time_discount=False,
                 amp=1e-2, reward_scaling=1):
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.f_c = f_c / GHz
        self.delta_time = GHz / f_c # nanoseconds
        self.amp = amp
        self.qubit = qubit
        self.reward_scaling=reward_scaling
        self.nmax = nmax
        self.fock_space_size = fock_space_size
        self.Zr = Zr
        self.extFlux = extFlux
        self.extVoltage = extVoltage
        self.temp = temp
        self.nn = nn
        self.time_discount = time_discount

        # frequency of unloaded resonator
        self.omega = omega
        # it's Q-factor
        self.Q = Q
        # resonator relaxation parameter
        self.gamma = self.omega * 0.69 / self.Q

        # qubit energy relaxation time
        self.T1 = T1

        # effective lumped capacitance of resonator
        self.Cr = 1 / (self.omega * self.Zr * GHz)
        
        C = self.qubit.Cq + self.qubit.Cg + self.qubit.Cx
        Ic =  self.qubit.Ic * abs(cos(pi * self.extFlux))
        Ej = (Ic * Fi0) / (hplank * GHz)
        Ec = (e**2 / (2 * C)) / (hplank * GHz)
        ng = self.extVoltage * self.qubit.Cx / (2 * e) # for now assume that extVoltage == 0
        energy_levels = [Ek_func(i * 2, Ec, Ej) for i in range(self.fock_space_size)[::-1]]
        energy_levels = np.array(energy_levels)
        # print(energy_levels)
        
        # energy difference between first and second energy level of qubit
        self.epsilon = energy_levels[-2] - energy_levels[-1]# self._init_transition_energy()
        # print('Epsilon', self.epsilon)
        
        
        # qubit-resonator couping constant
        self.g = self.qubit.Cg / (2 * sqrt(self.qubit.Cq * self.Cr)) * sqrt(self.omega * self.epsilon)
        # print('g', self.g)

        self._init_operators()

        # generic part of hamiltonian that represent non-interacting behaviour
        # of two parts of system: resonator and qubit
        # hplank is omitted because omega and epsilon have energy dimension
        if use_exact_energy_levels:
            H_transmon = qutip.Qobj(np.diag(energy_levels))
            self.H_transmon = tensor(qeye(self.nmax), H_transmon)
            self.H = 2 * pi * self.omega * (self.Aplus * self.Aminus) + 2 * pi * self.H_transmon
            # print(2 * pi * self.H_transmon)
        else:
            self.H = 2 * pi * self.omega * (self.Aplus * self.Aminus) +  2 * pi * 0.5 * self.epsilon * self.SigmaZ # + self.Qeye * (-19.11)
            # print(2 * pi * 0.5 * self.epsilon * self.SigmaZ)
        # interaction term
        # rabi model in rotating-wave approximation
        if RWA:
            self.H += self.g * (self.Aminus * self.SigmaP + self.Aplus * self.SigmaN)
        # full hamiltonian with both Jaynes-Cummings (JC) and anti-Jaynes-Cummings (AJC) terms
        else:
            self.H += self.g * (self.Aminus + self.Aplus) * self.SigmaX

        # print('H', self.H)

        # hamiltonian part representing qubit gate drive
        self.Ht = (e / (GHz * hplank)) * self.qubit.Cx * (self.qubit.Cq + self.qubit.Cg) / (self.qubit.Cq * self.qubit.Cg) * self.SigmaX

        # print(self.Ht)

        # print('Ht', self.Ht)
        # collapse operators
        self.cOps = [self.Aminus * sqrt(self.gamma * (1 + self.temp / self.omega)),
                     self.Aplus * sqrt(self.gamma * self.temp / self.omega),
                     self.SigmaP * sqrt(self.temp / self.epsilon / self.T1),
                     self.SigmaN * sqrt((1 + self.temp / self.epsilon) / self.T1),
                     self.SigmaZ * sqrt(self.gamma / 2)]

        # initial state of qubit in density matrix representation
        self.qubit_start = fock_dm(self.fock_space_size, 1)
        # initial state of resonatro in density matrix representation
        self.resonator_start = fock_dm(self.nmax, 0)
        self.qubit_resonator_start = tensor(self.resonator_start, self.qubit_start)
        # target state of qubit
        self.qubit_target = fock_dm(self.fock_space_size, 0)

        self.qubit_state = copy(self.qubit_start)
        self.resonator_state = copy(self.resonator_start)
        self.qubit_resonator_state = tensor(self.resonator_start, self.qubit_start)
        self.time_stamp = 0
        self.max_stamp = max_stamp
        self.action_steps = []
        self.options = Options(
            store_states=True,
            nsteps=8000,
            atol=1e-8,
            rtol=1e-6,
            num_cpus=3,
            order=3
        )
        self.seed()
        self.reset()

    def _init_operators(self):
        # lc resonator creation operator
        self.Aplus = tensor(create(self.nmax), qeye(self.fock_space_size))
        # annihilation operators
        self.Aminus = tensor(destroy(self.nmax), qeye(self.fock_space_size))

        self.sigmax = operators.jmat((self.fock_space_size - 1) / 2, 'x') * 2
        self.sigmaz = operators.jmat((self.fock_space_size - 1) / 2, 'z') * 2
        self.sigmap = operators.jmat((self.fock_space_size - 1) / 2, '+')
        self.sigmam = operators.jmat((self.fock_space_size - 1) / 2, '-')


        # pauli sigma matrix at x direction
        self.SigmaX = tensor(qeye(self.nmax), self.sigmax)
        # pauli sigma matrix at z direction
        self.SigmaZ = tensor(qeye(self.nmax), self.sigmaz)
        # creation operator
        self.SigmaP = tensor(qeye(self.nmax), self.sigmap)
        # annihilation operator
        self.SigmaN = tensor(qeye(self.nmax), self.sigmam)
        # particle number operator
        self.SigmaPopulation = tensor(qeye(self.nmax), self.sigmap * self.sigmam)
        self.Qeye = self.SigmaN = tensor(qeye(self.nmax), qeye(self.fock_space_size))

    def _init_transition_energy(self):
        """
        Calculate energy difference between zero level and first level of qubit
        """
        # (ev, phis, evec) = self.__qubitSpectrum(self.qp, 100, self.extFlux, self.extVoltage)
        Ic = self.qubit.Ic * abs(cos(np.pi * self.extFlux))
        Ej = (2 * Ic * Fi0 / 2) / (hplank * GHz)
        Ec = (e ** 2 / (2 * self.qubit.C)) / (hplank * GHz)
        return sqrt(8 * Ec * Ej) - Ec

    def reset(self):
        self.qubit_state = copy(self.qubit_start)
        self.resonator_state = copy(self.resonator_state)
        self.qubit_resonator_state = tensor(self.resonator_start, self.qubit_start)
        self.time_stamp = 0
        self.action_steps = [[0., 0.]]
        self.fidelity = 0

        return self.step([0.0, 0.0])[0] # take 0.0 action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        self.qubit_state

    def __state_data(self):
        tmp = self.qubit_state.data.toarray()
        real = tmp.real.flatten()
        img = tmp.imag.flatten()
        return np.hstack((real, img))

    def __qubitSpectrum(self, N=1000):

        step = 2 * pi / N

        C = self.qubit.Cq + self.qubit.Cg + self.qubit.Cx
        Ic =  self.qubit.Ic * abs(cos(pi*self.extFlux))
        Ej = (Ic * Fi0) / (hplank * GHz)
        Ec = (e**2 / (2 * C)) / (hplank * GHz)
        ng = self.extVoltage * self.qubit.Cx / (2 * e) # for now assume that extVoltage == 0

        phi = np.linspace(-pi, pi, N+1);
        phi = phi[0 : -1];

        def alpha(phi):
            return -4 * Ec
        def beta(phi):
            return 0
        def gamma(phi):
            return -cos(phi) * Ej

        diagCentr = np.zeros([N], dtype='complex')
        diagUp = np.zeros([N], dtype='complex')
        diagDown = np.zeros([N], dtype='complex')

        for i in range(N):
            diagCentr[i] = gamma(phi[i])- 2 * alpha(phi[i])/(step * step)
            diagUp[i] = alpha(phi[i]) / (step * step) + beta(phi[i]) / 2 / step
            diagDown[i] = alpha(phi[i]) / (step * step) - beta(phi[i]) / 2 / step

        phasefactor = np.exp(1j*ng*pi)

        sm = sparse.diags([[np.conj(phasefactor)*diagUp[-1]], diagDown[1:], diagCentr, diagUp[0: -1], [phasefactor*diagDown[1]]], [-N + 1, -1, 0, 1, N -1])
        sm = sm.toarray();
        (ev, evec) = np.linalg.eigh(sm)

        return ev, phi, evec

    def step(self, action):
        self.time_stamp += 1
        if self.time_stamp >= self.max_stamp:
            episode_over = True
        else:
            episode_over = False

        self.action_steps.append(action)
        new_fidelity, qubit_state, resonator_state, qubit_resonator_state, mesolve_result = self.__qubit_eval(np.array(self.action_steps))
        self.qubit_state = qubit_state
        self.resonator_state = resonator_state
        self.qubit_resonator_state = qubit_resonator_state
        state_data = self.__state_data()
        dum_action = dummyExternalDrive2D(self.delta_time * self.time_stamp, self.epsilon, self.f_c)
        observable_data = [
            # physical measures
            mesolve_result.expect[0][-1],
            mesolve_result.expect[1][-1],
            mesolve_result.expect[2][-1],
            mesolve_result.expect[3][-1],
            mesolve_result.expect[4][-1],
            # self-defined additional measures which could help agent to train
            self.fidelity,
            dum_action[0],
            dum_action[1],
            action[0],
            action[1]
        ]
        reward = new_fidelity - self.fidelity
        self.fidelity = new_fidelity
        
        reward = self.reward_scaling * reward
        if self.time_discount: reward /= sqrt(self.time_stamp)
        return observable_data, reward, episode_over, {'fidelity': self.fidelity}

    @staticmethod
    def _optimal_control(t, args):
        return args['amp'] * args['action'][0] * np.sin(args['f_c'] * (args['time'] + t)) + \
               args['amp'] * args['action'][1] * np.cos(args['f_c'] * (args['time'] + t))

    def __qubit_eval(self, action_steps):
        result = mesolve(H=[self.H, [self.Ht, self._optimal_control]],
                         rho0=self.qubit_resonator_state,
                         tlist=[0, self.delta_time],
                         c_ops=self.cOps,
                         e_ops=[self.Aplus + self.Aminus,
                                self.Aplus * self.Aminus,
                                self.SigmaZ,
                                self.SigmaX,
                                self.SigmaPopulation],
                         args={
                             'action': action_steps[-1], 'f_c': self.f_c, 
                             'time': self.time_stamp * self.delta_time,
                             'amp': self.amp,
                         },
                         options=self.options)


        qubit_state = result.states[-1].ptrace(1)
        resonator_state = result.states[-1].ptrace(0)
        qubit_resonator_state = result.states[-1]
        loss = fidelity(qubit_state, self.qubit_target)
        return loss, qubit_state, resonator_state, qubit_resonator_state, result

class EnvRNN(gym.Env):
    def __init__(self, env, seq_length):
        self.env = env
        self.seq_length = seq_length
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        

    def reset(self):
        obs = self.env.reset()
        self.seq = [obs] * self.seq_length
        return self.seq

    def step(self, action):
        obs, reward, episode_over, info = self.env.step(action)
        self.seq.append(obs)
        self.seq = self.seq[-self.seq_length:]
        return self.seq, reward, episode_over, info