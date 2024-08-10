'''
Solves the LPN equations of Regazzoni 2022 and Salvador 2023
Performs a full LPN simulation; that is, uses a time-varying elastance model
of the ventricles, instead of being coupled to a 3D finite element model of the
ventricles.
'''

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt
import os
import sys
from cycler import cycler
import warnings

# Initialize the dictionary that stores max_phi for different parameter combinations
# for double_Hill model. This is used to normalize phi in the double Hill function.
max_phi_dict = {}

def activation(t, parameters, model='double_cosine'):
    '''
    Time-varying activation function for the ventricles or atria.
    
    model:
        -'double_cosine': Function is composed of two half-cosine functions, one 
        which increases from 0 to 1 over the contraction phase, and one which 
        decreases from 1 to 0 over the relaxation phase.
        -'double_Hill': Function is composed of two Hill functions, one which
        increases from 0 to 1 over the contraction phase, and one which decreases
        from 1 to 0 over the relaxation phase.

    ARGS:
        - t: Time
        - parameters: Dictionary containing the following keys:
            - if model == 'double_cosine':
                - t_C: Time of contraction start
                - T_C: Duration of contraction
                - T_R: Duration of relaxation
                - T_HB: Heartbeat period
            - if model == 'double_Hill':
                - t_C: Time of contraction start
                - tau1: Time constant for the first Hill function
                - tau2: Time constant for the second Hill function
                - m1: Hill coefficient for the first Hill function
                - m2: Hill coefficient for the second Hill function
        - model: 'double_cosine' or 'double_Hill'
    
    RETURNS:
        - Activation value
    '''

    if model == 'double_cosine':
        t_C = parameters['t_C']
        T_C = parameters['T_C']
        T_R = parameters['T_R']
        T_HB = parameters['T_HB']

        t_R = t_C + T_C
        pi_over_T_C = np.pi / T_C
        pi_over_T_R = np.pi / T_R

        mod_t_C = np.mod(t - t_C, T_HB)
        mod_t_R = np.mod(t - t_R, T_HB)

        term1 = 0.5 * (1 - np.cos(pi_over_T_C * mod_t_C)) * (0 <= mod_t_C) * (mod_t_C < T_C)
        term2 = 0.5 * (1 + np.cos(pi_over_T_R * mod_t_R)) * (0 <= mod_t_R) * (mod_t_R < T_R)

        phi = term1 + term2

        return phi
    elif model == 'double_Hill':
        global max_phi_dict # for double_Hill model

        t_C = parameters['t_C']
        tau1 = parameters['tau1']
        tau2 = parameters['tau2']
        m1 = parameters['m1']
        m2 = parameters['m2']
        T_HB = parameters['T_HB']
    
        # Create a string representation of the parameters
        parameters_str = str(parameters)

        # Check if max_phi for these parameters is already in the dictionary
        if parameters_str not in max_phi_dict:
            # Compute the maximum value of phi in [0, T_HB]
            t_values = np.linspace(0, T_HB, 1000)
            g1_values = (np.mod(t_values - t_C, T_HB) / tau1) ** m1
            g2_values = (np.mod(t_values - t_C, T_HB) / tau2) ** m2
            phi_values = g1_values/(1+g1_values) * (1/(1+g2_values))
            max_phi = max(phi_values)

            # Store max_phi in the dictionary
            max_phi_dict[parameters_str] = max_phi
        else:
            # Retrieve max_phi from the dictionary
            max_phi = max_phi_dict[parameters_str]

        # Compute phi at time t
        g1 = (np.mod(t - t_C, T_HB) / tau1) ** m1
        g2 = (np.mod(t - t_C, T_HB) / tau2) ** m2
        phi = g1/(1+g1) * (1/(1+g2))

        # Normalize phi
        phi = phi / max_phi

        return phi

class Simulation:
    def __init__(self, sim_parameters):
        self.parameters = {}
        self.n_cardiac_cyc = sim_parameters[0]
        self.dt = sim_parameters[1]
        self.save_last_n_cardiac_cycles = sim_parameters[2]

    def A_LV(self, t):
        '''
        Time-varying activation function for the left ventricle.
        '''
        if self.parameters['activation_model'] == 'double_cosine':
            parameters = {
                't_C': self.parameters['timing_parameters_double_cosine']['t_C_LV'],
                'T_C': self.parameters['timing_parameters_double_cosine']['T_C_LV'],
                'T_R': self.parameters['timing_parameters_double_cosine']['T_R_LV'],
                'T_HB': self.parameters['T_HB']
            }
            return activation(t, parameters, model='double_cosine')

        elif self.parameters['activation_model'] == 'double_Hill':
            parameters = {
                't_C': self.parameters['timing_parameters_double_Hill']['t_C_LV'],
                'tau1': self.parameters['timing_parameters_double_Hill']['tau1_LV'],
                'tau2': self.parameters['timing_parameters_double_Hill']['tau2_LV'],
                'm1': self.parameters['timing_parameters_double_Hill']['m1_LV'],
                'm2': self.parameters['timing_parameters_double_Hill']['m2_LV'],
                'T_HB': self.parameters['T_HB']
            }
            return activation(t, parameters, model='double_Hill')

    def A_RV(self, t):
        '''
        Time-varying activation function for the right ventricle.
        '''
        if self.parameters['activation_model'] == 'double_cosine':
            parameters = {
                't_C': self.parameters['timing_parameters_double_cosine']['t_C_RV'],
                'T_C': self.parameters['timing_parameters_double_cosine']['T_C_RV'],
                'T_R': self.parameters['timing_parameters_double_cosine']['T_R_RV'],
                'T_HB': self.parameters['T_HB']
            }
            return activation(t, parameters, model='double_cosine')

        elif self.parameters['activation_model'] == 'double_Hill':
            parameters = {
                't_C': self.parameters['timing_parameters_double_Hill']['t_C_RV'],
                'tau1': self.parameters['timing_parameters_double_Hill']['tau1_RV'],
                'tau2': self.parameters['timing_parameters_double_Hill']['tau2_RV'],
                'm1': self.parameters['timing_parameters_double_Hill']['m1_RV'],
                'm2': self.parameters['timing_parameters_double_Hill']['m2_RV'],
                'T_HB': self.parameters['T_HB']
            }
            return activation(t, parameters, model='double_Hill')

    def A_LA(self, t):
        '''
        Time-varying activation function for the left atrium.
        '''
        if self.parameters['activation_model'] == 'double_cosine':
            parameters = {
                't_C': self.parameters['timing_parameters_double_cosine']['t_C_LA'],
                'T_C': self.parameters['timing_parameters_double_cosine']['T_C_LA'],
                'T_R': self.parameters['timing_parameters_double_cosine']['T_R_LA'],
                'T_HB': self.parameters['T_HB']
            }
            return activation(t, parameters, model='double_cosine')

        elif self.parameters['activation_model'] == 'double_Hill':
            parameters = {
                't_C': self.parameters['timing_parameters_double_Hill']['t_C_LA'],
                'tau1': self.parameters['timing_parameters_double_Hill']['tau1_LA'],
                'tau2': self.parameters['timing_parameters_double_Hill']['tau2_LA'],
                'm1': self.parameters['timing_parameters_double_Hill']['m1_LA'],
                'm2': self.parameters['timing_parameters_double_Hill']['m2_LA'],
                'T_HB': self.parameters['T_HB']
            }
            return activation(t, parameters, model='double_Hill')

    def A_RA(self, t):
        '''
        Time-varying activation function for the right atrium.
        '''
        if self.parameters['activation_model'] == 'double_cosine':
            parameters = {
                't_C': self.parameters['timing_parameters_double_cosine']['t_C_RA'],
                'T_C': self.parameters['timing_parameters_double_cosine']['T_C_RA'],
                'T_R': self.parameters['timing_parameters_double_cosine']['T_R_RA'],
                'T_HB': self.parameters['T_HB'],
            }
            return activation(t, parameters, model='double_cosine')

        elif self.parameters['activation_model'] == 'double_Hill':
            parameters = {
                't_C': self.parameters['timing_parameters_double_Hill']['t_C_RA'],
                'tau1': self.parameters['timing_parameters_double_Hill']['tau1_RA'],
                'tau2': self.parameters['timing_parameters_double_Hill']['tau2_RA'],
                'm1': self.parameters['timing_parameters_double_Hill']['m1_RA'],
                'm2': self.parameters['timing_parameters_double_Hill']['m2_RA'],
                'T_HB': self.parameters['T_HB']
            }
            return activation(t, parameters, model='double_Hill')

    def R_V(self, p1, p2):
        '''
        Valve resistance function. If p1 >= p2, return R_min, else return R_max.
        '''
        #R_V = lambda p1, p2: np.where(p1 >= p2, R_min, R_max)
        return self.parameters['R_min'] if p1 >= p2 else self.parameters['R_max']   # Faster than np.where for scalars


    def get_parameters(self):
        bn = 3

        # Parameters that change
        self.parameters['BPM'] = 87 * np.random.uniform(0.7, 1.5)
        self.parameters['T_HB'] = 60 / self.parameters['BPM']
        self.parameters['PR_interval'] = self.parameters['T_HB'] * 0.2639 * np.random.uniform(0.7, 1.2)
        # self.parameters['QRS_duration'] = self.parameters['T_HB'] * 0.1276 * np.random.uniform(0.9, 1.1)
        # self.parameters['QT_interval'] = self.parameters['T_HB'] * 0.5568 * np.random.uniform(0.9, 1.1)
        self.parameters['EMD'] = self.parameters['T_HB'] * 0.03625 * np.random.uniform(0.8, 1.2)

        self.parameters['R_AR_SYS'] = 0.42
        self.parameters['C_AR_SYS'] = 1.403
        self.parameters['R_VEN_SYS'] = 0.228
        self.parameters['C_VEN_SYS'] = 60.0
        self.parameters['R_AR_PUL'] = 0.032
        self.parameters['C_AR_PUL'] = 10.0
        self.parameters['R_VEN_PUL'] = 0.035
        self.parameters['C_VEN_PUL'] = 16.0

        self.parameters['chamber_model_linear'] = {}
        self.parameters['chamber_model_linear']['E_LV_act'] = 12.5 * (bn ** np.random.uniform(-1.5, 1.5))
        self.parameters['chamber_model_linear']['E_LV_pas'] = 0.075 * (bn ** np.random.uniform(-1.5, 1.5))
        self.parameters['chamber_model_linear']['E_RV_act'] = 1.526 * (bn ** np.random.uniform(-1.5, 1.5))
        self.parameters['chamber_model_linear']['E_RV_pas'] = 0.0375 * (bn ** np.random.uniform(-1.5, 1.5))
        self.parameters['chamber_model_linear']['E_LA_act'] = 0.15 * (bn ** np.random.uniform(-1.5, 1.5))
        self.parameters['chamber_model_linear']['E_LA_pas'] = 0.169 * (bn ** np.random.uniform(-1.5, 1.5))
        self.parameters['chamber_model_linear']['E_RA_act'] = 0.15 * (bn ** np.random.uniform(-1.5, 1.5))
        self.parameters['chamber_model_linear']['E_RA_pas'] = 0.075 * (bn ** np.random.uniform(-1.5, 1.5))
        self.parameters['chamber_model_linear']['V0_LV'] = 31.378 * (bn ** np.random.uniform(-1, 1))
        self.parameters['chamber_model_linear']['V0_RV'] = 73.19 * (bn ** np.random.uniform(-1, 1))
        self.parameters['chamber_model_linear']['V0_LA'] = 30.325 * (bn ** np.random.uniform(-1, 1))
        self.parameters['chamber_model_linear']['V0_RA'] = 41.482 * (bn ** np.random.uniform(-1, 1))

        ## Parameters that does not change
        self.parameters['t_final'] = self.parameters['T_HB'] * self.n_cardiac_cyc
        # self.parameters['n_timesteps'] = int(self.parameters['t_final'] / self.dt)
        self.parameters['n_timesteps'] = int(self.n_cardiac_cyc / self.dt)
        self.parameters['save_last_n_timesteps'] = int(
            self.save_last_n_cardiac_cycles / self.n_cardiac_cyc * self.parameters['n_timesteps'])

        # Chamber model: 'linear' or 'corsini2014' or 'zhang2023
        self.parameters['chamber_model'] = 'linear'
        # Activation model: 'double_cosine' or 'double_Hill'
        self.parameters['activation_model'] = 'double_Hill'

        # Intermediate timing parameters
        self.parameters['t_start_A'] = self.parameters['EMD']
        # self.parameters['t_end_A'] = self.parameters['PR_interval'] + self.parameters['QRS_duration']
        self.parameters['t_start_V'] = self.parameters['PR_interval'] + self.parameters['EMD']
        # self.parameters['t_end_V'] = self.parameters['PR_interval'] + self.parameters['QT_interval']

        self.parameters['timing_parameters_double_Hill'] = {}

        self.parameters['timing_parameters_double_Hill']['m1_LA'] = 1.32
        self.parameters['timing_parameters_double_Hill']['m2_LA'] = 13.1
        self.parameters['timing_parameters_double_Hill']['tau1_LA'] = self.parameters['T_HB'] * 0.110
        self.parameters['timing_parameters_double_Hill']['tau2_LA'] = self.parameters['T_HB'] * 0.210
        self.parameters['timing_parameters_double_Hill']['t_C_LA'] = self.parameters['t_start_A']

        self.parameters['timing_parameters_double_Hill']['m1_RA'] = self.parameters['timing_parameters_double_Hill']['m1_LA']
        self.parameters['timing_parameters_double_Hill']['m2_RA'] = self.parameters['timing_parameters_double_Hill']['m2_LA']
        self.parameters['timing_parameters_double_Hill']['tau1_RA'] = self.parameters['timing_parameters_double_Hill']['tau1_LA']
        self.parameters['timing_parameters_double_Hill']['tau2_RA'] = self.parameters['timing_parameters_double_Hill']['tau2_LA']
        self.parameters['timing_parameters_double_Hill']['t_C_RA'] = self.parameters['timing_parameters_double_Hill']['t_C_LA']

        self.parameters['timing_parameters_double_Hill']['m1_LV'] = 1.32
        self.parameters['timing_parameters_double_Hill']['m2_LV'] = 27.4
        self.parameters['timing_parameters_double_Hill']['tau1_LV'] = self.parameters['T_HB'] * 0.269
        self.parameters['timing_parameters_double_Hill']['tau2_LV'] = self.parameters['T_HB'] * 0.452
        self.parameters['timing_parameters_double_Hill']['t_C_LV'] = self.parameters['t_start_V']

        self.parameters['timing_parameters_double_Hill']['m1_RV'] = self.parameters['timing_parameters_double_Hill']['m1_LV']
        self.parameters['timing_parameters_double_Hill']['m2_RV'] = self.parameters['timing_parameters_double_Hill']['m2_LV']
        self.parameters['timing_parameters_double_Hill']['tau1_RV'] = self.parameters['timing_parameters_double_Hill']['tau1_LV']
        self.parameters['timing_parameters_double_Hill']['tau2_RV'] = self.parameters['timing_parameters_double_Hill']['tau2_LV']
        self.parameters['timing_parameters_double_Hill']['t_C_RV'] = self.parameters['timing_parameters_double_Hill']['t_C_LV']

        self.parameters['L_AR_SYS'] = 0.005
        self.parameters['Z_AR_SYS'] = 0
        self.parameters['L_VEN_SYS'] = 0.0005
        self.parameters['L_AR_PUL'] = 0.0005
        self.parameters['Z_AR_PUL'] = 0
        self.parameters['L_VEN_PUL'] = 0.0005

        self.parameters['R_min'] = 0.0075
        self.parameters['R_max'] = 7.5

        # Initial conditions
        self.parameters['V_LA_0'] = 59.79   # mL
        self.parameters['V_LV_0'] = 90.31   # mL
        self.parameters['V_RA_0'] = 75.77   # mL
        self.parameters['V_RV_0'] = 130.42  # mL
        self.parameters['p_LA_0'] = 4.987   # mmHg
        self.parameters['p_LV_0'] = 4.425   # mmHg
        self.parameters['p_RA_0'] = 2.57    # mmHg
        self.parameters['p_RV_0'] = 2.14    # mmHg
        self.parameters['p_AR_SYS_0'] = 63.44  # mmHg
        self.parameters['p_VEN_SYS_0'] = 29.66  # mmHg
        self.parameters['p_AR_PUL_0'] = 13.81  # mmHg
        self.parameters['p_VEN_PUL_0'] = 11.36  # mmHg
        self.parameters['Q_AR_SYS_0'] = 92.76  # mL s^-1
        self.parameters['Q_VEN_SYS_0'] = 119.04  # mL s^-1
        self.parameters['Q_AR_PUL_0'] = 77.02  # mL s^-1
        self.parameters['Q_VEN_PUL_0'] = 187.58  # mL s^-1


    def record_input(self):

        combined_input = np.array([ self.parameters['chamber_model_linear']['E_LV_act'],
                                    self.parameters['chamber_model_linear']['E_LV_pas'],
                                    self.parameters['chamber_model_linear']['E_RV_act'],
                                    self.parameters['chamber_model_linear']['E_RV_pas'],
                                    self.parameters['chamber_model_linear']['E_LA_act'],
                                    self.parameters['chamber_model_linear']['E_LA_pas'],
                                    self.parameters['chamber_model_linear']['E_RA_act'],
                                    self.parameters['chamber_model_linear']['E_RA_pas'],
                                    self.parameters['chamber_model_linear']['V0_LV'],
                                    self.parameters['chamber_model_linear']['V0_RV'],
                                    self.parameters['chamber_model_linear']['V0_LA'],
                                    self.parameters['chamber_model_linear']['V0_RA']
                                   ])

        return combined_input

    def regazzoni_2022_lpn_ode(self, t, y):
        '''
        The ODE system in Kerckhoffs 2007 (Appendices A and B)

        ARGS:
            - t: Time
            - y: LPN state vector
        '''

        # Unpack necessary parameters into variables

        # Systemic circulation
        R_AR_SYS = self.parameters['R_AR_SYS']  # mmHg s mL^-1
        C_AR_SYS = self.parameters['C_AR_SYS']  # mL mmHg^-1
        L_AR_SYS = self.parameters['L_AR_SYS']  # mmHg s^2 mL^-1

        R_VEN_SYS = self.parameters['R_VEN_SYS']  # mmHg s mL^-1
        C_VEN_SYS = self.parameters['C_VEN_SYS']  # mL mmHg^-1
        L_VEN_SYS = self.parameters['L_VEN_SYS']  # mmHg s^2 mL^-1

        # Pulmonary circulation
        R_AR_PUL = self.parameters['R_AR_PUL']  # mmHg s mL^-1
        C_AR_PUL = self.parameters['C_AR_PUL']  # mL mmHg^-1
        L_AR_PUL = self.parameters['L_AR_PUL']  # mmHg s^2 mL^-1

        R_VEN_PUL = self.parameters['R_VEN_PUL']  # mmHg s mL^-1
        C_VEN_PUL = self.parameters['C_VEN_PUL']  # mL mmHg^-1
        L_VEN_PUL = self.parameters['L_VEN_PUL']  # mmHg s^2 mL^-1

        # Unpack variables from state vector y
        V_LA = y[0]         # mL; Left atrial volume
        V_LV = y[1]         # mL; Left ventricular volume
        V_RA = y[2]         # mL; Right atrial volume
        V_RV = y[3]         # mL; Right ventricular volume
        p_AR_SYS = y[4]     # mmHg; Systemic arterial pressure
        p_VEN_SYS = y[5]    # mmHg; Systemic venous pressure
        p_AR_PUL = y[6]     # mmHg; Pulmonary arterial pressure
        p_VEN_PUL = y[7]    # mmHg; Pulmonary venous pressure
        Q_AR_SYS = y[8]     # mL/s; Systemic arterial flow
        Q_VEN_SYS = y[9]    # mL/s; Systemic venous flow
        Q_AR_PUL = y[10]    # mL/s; Pulmonary arterial flow
        Q_VEN_PUL = y[11]   # mL/s; Pulmonary venous flow

        # Cardiac pressures
        if self.parameters['chamber_model'] == 'linear':
            chamber_model_parameters = self.parameters['chamber_model_linear']

            # Passive elastances
            E_LV_pas = chamber_model_parameters['E_LV_pas']  # mmHg mL^-1
            E_LA_pas = chamber_model_parameters['E_LA_pas']  # mmHg mL^-1
            E_RV_pas = chamber_model_parameters['E_RV_pas']  # mmHg mL^-1
            E_RA_pas = chamber_model_parameters['E_RA_pas']  # mmHg mL^-1

            # Active elastances
            E_LV_act = chamber_model_parameters['E_LV_act']  # mmHg mL^-1
            E_LA_act = chamber_model_parameters['E_LA_act']  # mmHg mL^-1
            E_RV_act = chamber_model_parameters['E_RV_act']  # mmHg mL^-1
            E_RA_act = chamber_model_parameters['E_RA_act']  # mmHg mL^-1

            # Atrial rest volumes
            V0_LA = chamber_model_parameters['V0_LA']  # mL
            V0_RA = chamber_model_parameters['V0_RA']  # mL

            # Ventricular rest volumes
            V0_RV = chamber_model_parameters['V0_RV']  # mL
            V0_LV = chamber_model_parameters['V0_LV']  # mL

            # Compute pressure values
            p_LV = (E_LV_pas + E_LV_act * self.A_LV(t)) * (V_LV - V0_LV)
            p_LA = (E_LA_pas + E_LA_act * self.A_LA(t)) * (V_LA - V0_LA)
            p_RV = (E_RV_pas + E_RV_act * self.A_RV(t)) * (V_RV - V0_RV)
            p_RA = (E_RA_pas + E_RA_act * self.A_RA(t)) * (V_RA - V0_RA)

        # Flow rates. For valve resistance, if p1 --|>-- p2, R_V(p1, p2)
        Q_MV = (p_LA - p_LV) / self.R_V(p_LA, p_LV)
        Q_AV = (p_LV - p_AR_SYS) / self.R_V(p_LV, p_AR_SYS)
        Q_TV = (p_RA - p_RV) / self.R_V(p_RA, p_RV)
        Q_PV = (p_RV - p_AR_PUL) / self.R_V(p_RV, p_AR_PUL)

        # The ODEs for volumes
        dV_LA = Q_VEN_PUL - Q_MV
        dV_LV = Q_MV - Q_AV
        dV_RA = Q_VEN_SYS - Q_TV
        dV_RV = Q_TV - Q_PV

        dp_AR_SYS = 1/C_AR_SYS * (Q_AV - Q_AR_SYS)
        dp_VEN_SYS = 1/C_VEN_SYS * (Q_AR_SYS - Q_VEN_SYS)
        dp_AR_PUL = 1/C_AR_PUL * (Q_PV - Q_AR_PUL)
        dp_VEN_PUL = 1/C_VEN_PUL * (Q_AR_PUL - Q_VEN_PUL)

        dQ_AR_SYS = R_AR_SYS/L_AR_SYS * (-Q_AR_SYS - (p_VEN_SYS - p_AR_SYS) / R_AR_SYS)
        dQ_VEN_SYS = R_VEN_SYS/L_VEN_SYS * (-Q_VEN_SYS - (p_RA - p_VEN_SYS) / R_VEN_SYS)
        dQ_AR_PUL = R_AR_PUL/L_AR_PUL * (-Q_AR_PUL - (p_VEN_PUL - p_AR_PUL) / R_AR_PUL)
        dQ_VEN_PUL = R_VEN_PUL/L_VEN_PUL * (-Q_VEN_PUL - (p_LA - p_VEN_PUL) / R_VEN_PUL)


        # Package derivatives
        return np.array([dV_LA, dV_LV, dV_RA, dV_RV,
                         dp_AR_SYS, dp_VEN_SYS, dp_AR_PUL, dp_VEN_PUL,
                         dQ_AR_SYS, dQ_VEN_SYS, dQ_AR_PUL, dQ_VEN_PUL])


    def rk4 (self, dydt, tspan, y0, n ):

        #*****************************************************************************80
        #
        ## rk4() approximates the solution to an ODE using the RK4 method.
        #
        #  Licensing:
        #
        #    This code is distributed under the GNU LGPL license.
        #
        #  Modified:
        #
        #    22 April 2020
        #
        #  Author:
        #
        #    John Burkardt
        #
        #  Input:
        #
        #    function dydt: points to a function that evaluates the right
        #    hand side of the ODE.
        #
        #    real tspan[2]: contains the initial and final times.
        #
        #    real y0[m]: an array containing the initial condition.
        #
        #    integer n: the number of steps to take.
        #
        #  Output:
        #
        #    real t[n+1], y[n+1,m]: the times and solution values.
        #

        if ( np.ndim ( y0 ) == 0 ):
            m = 1
        else:
            m = len ( y0 )

        tfirst = tspan[0]
        tlast = tspan[1]
        dt = ( tlast - tfirst ) / n
        t = np.zeros ( n + 1 )
        y = np.zeros ( [ n + 1, m ] )
        t[0] = tspan[0]
        y[0,:] = y0

        for i in range ( 0, n ):

            f1 = dydt ( t[i],            y[i,:] )
            f2 = dydt ( t[i] + dt / 2.0, y[i,:] + dt * f1 / 2.0 )
            f3 = dydt ( t[i] + dt / 2.0, y[i,:] + dt * f2 / 2.0 )
            f4 = dydt ( t[i] + dt,       y[i,:] + dt * f3 )

            t[i+1] = t[i] + dt
            y[i+1,:] = y[i,:] + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0

        return t, y

    def integrate(self):
        '''
        Integrates the kerckhoffs_2007 LPN. Output results to .npz, .txt, and plots.
        LPN parameters defined in kerckhoffs2007_parameters_modified.py
        General sim parameters defined below.
        '''

        # Unpack initial conditions
        V_LA_0 = self.parameters['V_LA_0']
        V_LV_0 = self.parameters['V_LV_0']
        V_RA_0 = self.parameters['V_RA_0']
        V_RV_0 = self.parameters['V_RV_0']
        p_AR_SYS_0 = self.parameters['p_AR_SYS_0']
        p_VEN_SYS_0 = self.parameters['p_VEN_SYS_0']
        p_AR_PUL_0 = self.parameters['p_AR_PUL_0']
        p_VEN_PUL_0 = self.parameters['p_VEN_PUL_0']
        Q_AR_SYS_0 = self.parameters['Q_AR_SYS_0']
        Q_VEN_SYS_0 = self.parameters['Q_VEN_SYS_0']
        Q_AR_PUL_0 = self.parameters['Q_AR_PUL_0']
        Q_VEN_PUL_0 = self.parameters['Q_VEN_PUL_0']

        # Unpack other parameters
        R_AR_SYS = self.parameters['R_AR_SYS']
        C_AR_SYS = self.parameters['C_AR_SYS']
        L_AR_SYS = self.parameters['L_AR_SYS']
        Z_AR_SYS = self.parameters['Z_AR_SYS']
        R_VEN_SYS = self.parameters['R_VEN_SYS']
        C_VEN_SYS = self.parameters['C_VEN_SYS']
        L_VEN_SYS = self.parameters['L_VEN_SYS']
        R_AR_PUL = self.parameters['R_AR_PUL']
        C_AR_PUL = self.parameters['C_AR_PUL']
        L_AR_PUL = self.parameters['L_AR_PUL']
        Z_AR_PUL = self.parameters['Z_AR_PUL']
        R_VEN_PUL = self.parameters['R_VEN_PUL']
        C_VEN_PUL = self.parameters['C_VEN_PUL']
        L_VEN_PUL = self.parameters['L_VEN_PUL']
        R_min = self.parameters['R_min']
        R_max = self.parameters['R_max']
        T_HB = self.parameters['T_HB']

        t_final = self.parameters['t_final']
        n_timesteps = self.parameters['n_timesteps']
        save_last_n_timesteps = self.parameters['save_last_n_timesteps']


        # Initial condition
        y_0 = np.array([V_LA_0, V_LV_0, V_RA_0, V_RV_0,
                        p_AR_SYS_0, p_VEN_SYS_0, p_AR_PUL_0, p_VEN_PUL_0,
                        Q_AR_SYS_0, Q_VEN_SYS_0, Q_AR_PUL_0, Q_VEN_PUL_0])

        # Integrate ODE system
        # t1 = time.time()
        t, y = self.rk4(self.regazzoni_2022_lpn_ode, [0, t_final], y_0, n_timesteps)
        # print(f"{time.time() - t1:.2f}")

        t = t[-save_last_n_timesteps:]
        V_LA = y[-save_last_n_timesteps:,0]
        V_LV = y[-save_last_n_timesteps:,1]
        V_RA = y[-save_last_n_timesteps:,2]
        V_RV = y[-save_last_n_timesteps:,3]
        p_AR_SYS = y[-save_last_n_timesteps:,4]
        p_VEN_SYS = y[-save_last_n_timesteps:,5]
        p_AR_PUL = y[-save_last_n_timesteps:,6]
        p_VEN_PUL = y[-save_last_n_timesteps:,7]
        Q_AR_SYS = y[-save_last_n_timesteps:,8]
        Q_VEN_SYS = y[-save_last_n_timesteps:,9]
        Q_AR_PUL = y[-save_last_n_timesteps:,10]
        Q_VEN_PUL = y[-save_last_n_timesteps:,11]

        # Compute cardiac pressures from volume solution
        if self.parameters['chamber_model'] == 'linear':
            chamber_model_parameters = self.parameters['chamber_model_linear']

            E_LV_act = chamber_model_parameters['E_LV_act']
            E_LV_pas = chamber_model_parameters['E_LV_pas']
            E_RV_act = chamber_model_parameters['E_RV_act']
            E_RV_pas = chamber_model_parameters['E_RV_pas']
            E_LA_act = chamber_model_parameters['E_LA_act']
            E_LA_pas = chamber_model_parameters['E_LA_pas']
            E_RA_act = chamber_model_parameters['E_RA_act']
            E_RA_pas = chamber_model_parameters['E_RA_pas']
            V0_LA = chamber_model_parameters['V0_LA']
            V0_RA = chamber_model_parameters['V0_RA']
            V0_RV = chamber_model_parameters['V0_RV']
            V0_LV = chamber_model_parameters['V0_LV']

            p_LV = (E_LV_pas + E_LV_act * self.A_LV(t)) * (V_LV - V0_LV)
            p_LA = (E_LA_pas + E_LA_act * self.A_LA(t)) * (V_LA - V0_LA)
            p_RV = (E_RV_pas + E_RV_act * self.A_RV(t)) * (V_RV - V0_RV)
            p_RA = (E_RA_pas + E_RA_act * self.A_RA(t)) * (V_RA - V0_RA)

        # Compute vascular volumes from pressure solution
        V_AR_SYS = p_AR_SYS * C_AR_SYS
        V_VEN_SYS = p_VEN_SYS * C_VEN_SYS
        V_AR_PUL = p_AR_PUL * C_AR_PUL
        V_VEN_PUL = p_VEN_PUL * C_VEN_PUL

        # Compute cardiac valve flows from pressure solution
        R_V = lambda p1, p2: np.where(p1 >= p2, R_min, R_max)
        Q_MV = (p_LA - p_LV) / R_V(p_LA, p_LV)
        Q_AV = (p_LV - p_AR_SYS) / R_V(p_LV, p_AR_SYS)
        Q_TV = (p_RA - p_RV) / R_V(p_RA, p_RV)
        Q_PV = (p_RV - p_AR_PUL) / R_V(p_RV, p_AR_PUL)

        # Compute total blood volume
        V_tot = V_LA + V_LV + V_RA + V_RV + V_AR_SYS + V_VEN_SYS + V_AR_PUL + V_VEN_PUL

        results_dict = {
            'time': t,
            'V_LA': V_LA,
            'V_LV': V_LV,
            'V_RA': V_RA,
            'V_RV': V_RV,
            'V_AR_SYS': V_AR_SYS,
            'V_VEN_SYS': V_VEN_SYS,
            'V_AR_PUL': V_AR_PUL,
            'V_VEN_PUL': V_VEN_PUL,
            'V_tot': V_tot,
            'p_LA': p_LA,
            'p_LV': p_LV,
            'p_RA': p_RA,
            'p_RV': p_RV,
            'p_AR_SYS': p_AR_SYS,
            'p_VEN_SYS': p_VEN_SYS,
            'p_AR_PUL': p_AR_PUL,
            'p_VEN_PUL': p_VEN_PUL,
            'Q_MV': Q_MV,
            'Q_AV': Q_AV,
            'Q_TV': Q_TV,
            'Q_PV': Q_PV,
            'Q_AR_SYS': Q_AR_SYS,
            'Q_VEN_SYS': Q_VEN_SYS,
            'Q_AR_PUL': Q_AR_PUL,
            'Q_VEN_PUL': Q_VEN_PUL,
            'R_MV': R_V(p_LA, p_LV),
            'R_AV': R_V(p_LV, p_AR_SYS),
            'R_TV': R_V(p_RA, p_RV),
            'R_PV': R_V(p_RV, p_AR_PUL),
            'A_LV': self.A_LV(t),
            'A_RV': self.A_RV(t),
            'A_LA': self.A_LA(t),
            'A_RA': self.A_RA(t)
        }

        return results_dict

    def save_results(self, results_dict, output_dir):
        '''
        Save results to .npz

        ARGS:
            - results_dict: Dictionary containing simulation results.
            - output_dir: Directory to save results.
        '''
        os.makedirs(output_dir, exist_ok=True)

        # Save all pressures to text file
        with open(os.path.join(output_dir, 'pressures.txt'), 'w') as f:
            f.write('time\tp_LA\tp_LV\tp_RA\tp_RV\tp_AR_SYS\tp_VEN_SYS\tp_AR_PUL\tp_VEN_PUL\n')
            for i in range(len(results_dict['time'])):
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    results_dict['time'][i],
                    results_dict['p_LA'][i],
                    results_dict['p_LV'][i],
                    results_dict['p_RA'][i],
                    results_dict['p_RV'][i],
                    results_dict['p_AR_SYS'][i],
                    results_dict['p_VEN_SYS'][i],
                    results_dict['p_AR_PUL'][i],
                    results_dict['p_VEN_PUL'][i],
                ))
        # Save all volumes to text file
        with open(os.path.join(output_dir, 'volumes.txt'), 'w') as f:
            f.write('time\tV_LA\tV_LV\tV_RA\tV_RV\tV_AR_SYS\tV_VEN_SYS\tV_AR_PUL\tV_VEN_PUL\tV_tot\n')
            for i in range(len(results_dict['time'])):
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    results_dict['time'][i],
                    results_dict['V_LA'][i],
                    results_dict['V_LV'][i],
                    results_dict['V_RA'][i],
                    results_dict['V_RV'][i],
                    results_dict['V_AR_SYS'][i],
                    results_dict['V_VEN_SYS'][i],
                    results_dict['V_AR_PUL'][i],
                    results_dict['V_VEN_PUL'][i],
                    results_dict['V_tot'][i],
                ))

        # Save all flows to text file
        with open(os.path.join(output_dir, 'flows.txt'), 'w') as f:
            f.write('time\tQ_MV\tQ_AV\tQ_TV\tQ_PV\tQ_AR_SYS\tQ_VEN_SYS\tQ_AR_PUL\tQ_VEN_PUL\n')
            for i in range(len(results_dict['time'])):
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    results_dict['time'][i],
                    results_dict['Q_MV'][i],
                    results_dict['Q_AV'][i],
                    results_dict['Q_TV'][i],
                    results_dict['Q_PV'][i],
                    results_dict['Q_AR_SYS'][i],
                    results_dict['Q_VEN_SYS'][i],
                    results_dict['Q_AR_PUL'][i],
                    results_dict['Q_VEN_PUL'][i],
                ))

        # Save cardiac activations to text file
        with open(os.path.join(output_dir, 'activations.txt'), 'w') as f:
            f.write('time\tA_LV\tA_RV\tA_LA\tA_RA\n')
            for i in range(len(results_dict['time'])):
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    results_dict['time'][i],
                    results_dict['A_LV'][i],
                    results_dict['A_RV'][i],
                    results_dict['A_LA'][i],
                    results_dict['A_RA'][i],
                ))
        
        # Save clinical metrics to text file. Print value and units
        with open(os.path.join(output_dir, 'clinical_metrics.txt'), 'w') as f:
            for key, value in results_dict['clinical_metrics'].items():
                f.write(f"{key}: {value['Value']} {value['Units']}\n")

        # Save cardiac pressure and volume to text file
        with open(os.path.join(output_dir, 'cardiac_PV.txt'), 'w') as f:
            f.write('time\tRR%\tV_LV\tp_LV\tV_LA\tp_LA\tV_RV\tp_RV\tV_RA\tp_RA\n')
            for i in range(len(results_dict['time'])):
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    results_dict['time'][i],
                    results_dict['RR%'][i],
                    results_dict['V_LV'][i],
                    results_dict['p_LV'][i],
                    results_dict['V_LA'][i],
                    results_dict['p_LA'][i],
                    results_dict['V_RV'][i],
                    results_dict['p_RV'][i],
                    results_dict['V_RA'][i],
                    results_dict['p_RA'][i],
                ))
            

    def find_closest_index(lst, value):
        return min(range(len(lst)), key = lambda index : abs(lst[index]-value))


    def calc_cardiac_phases(self, results_dict):
        '''
        Calculate timings of cardiac phases from results.

        Also, calculates RR interval percentage from time and PR_interval
        '''

        # Timestep size
        dt = np.diff(results_dict['time'])[0]

        # Compute heart rate (compute from LV volume results)
        V_LV_hat = fft(results_dict['V_LV'] - np.mean(results_dict['V_LV'])) # Compute FFT of LV volume minus mean
        freq = fftfreq(len(results_dict['time']), d=results_dict['time'][1]-results_dict['time'][0]) # Compute frequency axis
        f_max = freq[np.argmax(np.abs(V_LV_hat))] # Find frequency of maximum power
        results_dict['HR'] = f_max * 60 # Beats per minute

        # Compute cardiac cycle duration
        results_dict['T_HB'] = 1. / f_max # Seconds

        # Plot FFT of LV volume
        #N = len(results_dict['time'])
        #plt.plot(freq[0:N//2], np.abs(V_hat[0:N//2]))
        #plt.show()

        # Compute number of cardiac cycles in results
        n_cycles = int(np.floor(len(results_dict['time']) / (results_dict['T_HB'] / dt)))
        results_dict['n_cycles'] = n_cycles

        # Compute time derivatives of valve resistances
        dR_MV = np.gradient(results_dict['R_MV'], dt)
        dR_AV = np.gradient(results_dict['R_AV'], dt) 
        dR_TV = np.gradient(results_dict['R_TV'], dt)
        dR_PV = np.gradient(results_dict['R_PV'], dt) 

        # Plot valve resistances and their time derivatives
        #plt.figure()
        #plt.plot(results_dict['time'], results_dict['R_MV'], label='R_MV')
        #plt.plot(results_dict['time'], results_dict['R_AV'], label='R_AV')
        #plt.plot(results_dict['time'], results_dict['R_TV'], label='R_TV')
        #plt.plot(results_dict['time'], results_dict['R_PV'], label='R_PV')
        #plt.legend()
        #plt.savefig('valve_resistances.png')

        # Find timesteps and time of max/min dR/dt
        n_MV_close, _ = find_peaks(dR_MV)
        n_MV_open, _ = find_peaks(-dR_MV)
        t_MV_close = results_dict['time'][n_MV_close]
        t_MV_open = results_dict['time'][n_MV_open]

        n_AV_close, _ = find_peaks(dR_AV)
        n_AV_open, _ = find_peaks(-dR_AV)
        t_AV_close = results_dict['time'][n_AV_close]
        t_AV_open= results_dict['time'][n_AV_open]

        n_TV_close, _ = find_peaks(dR_TV)
        n_TV_open, _ = find_peaks(-dR_TV)
        t_TV_close= results_dict['time'][n_TV_close]
        t_TV_open= results_dict['time'][n_TV_open]

        n_PV_close, _ = find_peaks(dR_PV)
        n_PV_open, _ = find_peaks(-dR_PV)
        t_PV_close = results_dict['time'][n_PV_close]
        t_PV_open = results_dict['time'][n_PV_open]


        # Find timestep of start of atrial contraction. Start of atrial 
        # contraction marked by maximum in second derivative of atrial activation
        # (either elastance or active pressure)

        # Compute second derivative of atrial activation
        d2A_LA = np.gradient(np.gradient(results_dict['A_LA'], dt), dt)
        d2A_RA = np.gradient(np.gradient(results_dict['A_RA'], dt), dt)

        # Plot second derivative of atrial activation
        #plt.figure()
        #plt.plot(results_dict['time'], results_dict['A_LA'], label='A_LA')
        #plt.plot(results_dict['time'], d2A_LA/1e4, label='d2A_LA/10^4')
        #plt.legend()
        #plt.savefig('atrial_activation.png')

        # Find timesteps and time of max d2A_LA and d2A_RA.
        n_C_LA, _ = find_peaks(d2A_LA)  # Find peaks in signal
        prominences = peak_prominences(d2A_LA, n_C_LA)[0]   # Compute prominences of peaks
        n_C_LA = n_C_LA[prominences.argsort()[-n_cycles:]]  # Extract the first n_cycle peaks with the highest prominences
        t_C_LA = results_dict['time'][n_C_LA]

        n_C_RA, _ = find_peaks(d2A_RA)  # Find peaks in signal
        prominences = peak_prominences(d2A_RA, n_C_RA)[0]   # Compute prominences of peaks
        n_C_RA = n_C_RA[prominences.argsort()[-n_cycles:]]  # Extract the first n_cycle peaks with the highest prominences
        t_C_RA = results_dict['time'][n_C_RA]
    
        # Save results in results_dict
        results_dict['n_MV_close'] = n_MV_close
        results_dict['n_MV_open'] = n_MV_open
        results_dict['t_MV_close'] = t_MV_close
        results_dict['t_MV_open'] = t_MV_open

        results_dict['n_AV_close'] = n_AV_close
        results_dict['n_AV_open'] = n_AV_open
        results_dict['t_AV_close'] = t_AV_close
        results_dict['t_AV_open'] = t_AV_open

        results_dict['n_TV_close'] = n_TV_close
        results_dict['n_TV_open'] = n_TV_open
        results_dict['t_TV_close'] = t_TV_close
        results_dict['t_TV_open'] = t_TV_open

        results_dict['n_PV_close'] = n_PV_close
        results_dict['n_PV_open'] = n_PV_open
        results_dict['t_PV_close'] = t_PV_close
        results_dict['t_PV_open'] = t_PV_open

        results_dict['n_C_LA'] = n_C_LA
        results_dict['n_C_RA'] = n_C_RA
        results_dict['t_C_LA'] = t_C_LA
        results_dict['t_C_RA'] = t_C_RA


        # Calculate RR interval percentage
        t_cyc = np.mod(results_dict['time'], results_dict['T_HB']) # Time in cardiac cycle
        t_R_cyc = results_dict['parameters']['PR_interval'] # Time of R wave in cardiac cycle
        results_dict['RR%'] = np.mod(t_cyc - t_R_cyc, results_dict['T_HB']) / results_dict['T_HB'] * 100 # Percentage of RR interval

        # Plot RR interval percentage
        #plt.figure()
        #plt.plot(results_dict['time'], results_dict['RR%'])
        #plt.title('RR interval percentage')
        #plt.xlabel('Time (s)')
        #plt.ylabel('RR interval (%)')
        #plt.show()

    def plot_variables_vs_time(self, results_dict, variables, title, y_label, output_dir, output_file, subplots=False, phase_transitions = None, font_size=12):
        '''
        Function to plot results vs time and save to file.

        ARGS:
            - results_dict: Dictionary containing simulation results.
            - variables: List of variables to plot.
            - title: Title of plot.
            - y_label: Label of y-axis.
            - output_dir: Directory to save plots.
            - output_file: Name of output file.
            - subplots: If True, plot each variable in separate subplot.
            - phase_transitions: If present, plot vertical lines at cardiac phase transitions.
            - font_size: Font size of plot labels.

        RETURNS:
            - None
        '''

        # Store the original font size
        original_font_size = plt.rcParams['font.size']

        # Increase the font size
        plt.rcParams.update({'font.size': font_size})

        # Plot quantities vs time
        if subplots:
            fig, axs = plt.subplots(len(variables), 1, figsize=(8, 3*len(variables)))
            for i, var in enumerate(variables):
                axs[i].plot(results_dict['time'], results_dict[var])
                axs[i].set_title(var)
                axs[i].set_ylabel(y_label)
            axs[-1].set_xlabel('Time (s)')

        else:
            plt.figure()
            for var in variables:
                plt.plot(results_dict['time'], results_dict[var], label=var)
            plt.title(title)
            plt.xlabel('Time (s)')
            plt.ylabel(y_label)
            plt.legend()

        # Shade cardiac phases
        if phase_transitions:
            for phase_transition in phase_transitions:
                n_phase_transition = results_dict[phase_transition]
                for n in n_phase_transition:
                    plt.axvline(x=results_dict['time'][n], color='grey', linestyle='--', linewidth=0.5, alpha=0.5)


        # Plot RR interval percentage as second x-axis
        RR_tick_start = results_dict['time'][0] + results_dict['parameters']['PR_interval']
        RR_tick_end = results_dict['time'][0] + results_dict['parameters']['PR_interval'] + results_dict['T_HB']
        RR_tick_locations = np.linspace(RR_tick_start, RR_tick_end, 11)

        xlim = plt.gca().get_xlim()
        ax2 = plt.gca().twiny()
        ax2.set_xlim(xlim)
        ax2.set_xticks(RR_tick_locations)
        ax2.set_xticklabels(np.linspace(0, 100, 11).astype(int))
        ax2.set_xlabel('RR interval (%)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_file))
        plt.close()


        # Restore the original font size
        plt.rcParams.update({'font.size': original_font_size})

    def plot_pv_loops_without_phase(self, chamber, results_dict, font_size=12):
        '''
                Plot and save ventricular pressure volume loops with cardiac phases marked.
                '''

        # Store the original font size
        original_font_size = plt.rcParams['font.size']

        # Increase the font size
        plt.rcParams.update({'font.size': font_size})

        # Construct string based on chamber
        V = 'V_' + chamber
        p = 'p_' + chamber

        # Set color cycle

        plt.plot(results_dict[V], results_dict[p], '-', label=V)

    def plot_pv_loops_with_phases(self, chamber, results_dict, arrows = False, font_size=12):
        '''
        Plot and save ventricular pressure volume loops with cardiac phases marked.
        '''

        # Store the original font size
        original_font_size = plt.rcParams['font.size']

        # Increase the font size
        plt.rcParams.update({'font.size': font_size})

        # Construct string based on chamber
        V = 'V_' + chamber
        p = 'p_' + chamber

        # Set color cycle
        colors = ['yellowgreen', 'gold', 'indianred', 'mediumpurple', 'deepskyblue']
        plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

        # Label valve opening and closing points
        if chamber in ['LV', 'LA']:
            # Extract valve opening and closing timesteps and atrial contraction timesteps
            n_MV_close = results_dict['n_MV_close']
            n_MV_open = results_dict['n_MV_open']
            n_AV_close = results_dict['n_AV_close']
            n_AV_open = results_dict['n_AV_open']
            n_C_LA = results_dict['n_C_LA']

            # Check if number of transitions timesteps are equal (should be equal to number of cycles in results)
            condition = len(n_MV_close) == len(n_MV_open) == len(n_AV_close) == len(n_AV_open) == len(n_C_LA)
            if not condition:
                warnings.warn(f"Number of left heart cardiac phase timesteps not equal. "
                                f"MV_close: {len(n_MV_close)}, MV_open: {len(n_MV_open)}, "
                                f"AV_close: {len(n_AV_close)}, AV_open: {len(n_AV_open)}, C_LA: {len(n_C_LA)}"
                              )


            # Combine cardiac phase timesteps into single array
            n_close_open = np.sort(np.concatenate((n_MV_close, n_MV_open, n_AV_close, n_AV_open, n_C_LA)))

            # Add last timestep to n_close_open to ensure all timesteps are included
            n_close_open = np.append(n_close_open, -2) # -2 because we add 1 to the index in the loop below

            # Loop over cardiac phase timesteps
            for i in range(len(n_close_open)-1):
                # Reset color cycle after 5 iteration
                if i % 5 == 0:
                    plt.gca().set_prop_cycle(None)
                
                # Plot cardiac phases of PV loop in different colors
                plt.plot(results_dict[V][n_close_open[i]:n_close_open[i+1]+1], results_dict[p][n_close_open[i]:n_close_open[i+1]+1], '--')

                # Draw an arrow at the midpoint of each cardiac phase
                if arrows:
                    n_mid = (n_close_open[i] + n_close_open[i+1]) // 2
                    arrow_direction_x = results_dict[V][n_mid+1] - results_dict[V][n_mid]
                    arrow_direction_y = results_dict[p][n_mid+1] - results_dict[p][n_mid]

                    plt.arrow(results_dict[V][n_mid], results_dict[p][n_mid], 
                            arrow_direction_x, arrow_direction_y, 
                            head_width=2.0, fc='k', ec='k')

            # Plot markers at cardiac phase transitions
            plt.plot(results_dict[V][results_dict['n_MV_open']], results_dict[p][results_dict['n_MV_open']], 'ko', fillstyle='none', label='MV open')
            plt.plot(results_dict[V][results_dict['n_C_LA']], results_dict[p][results_dict['n_C_LA']], 'k*', fillstyle='none', label='Atr. cont.')
            plt.plot(results_dict[V][results_dict['n_MV_close']], results_dict[p][results_dict['n_MV_close']], 'kx', fillstyle='none', label='MV close')
            plt.plot(results_dict[V][results_dict['n_AV_open']], results_dict[p][results_dict['n_AV_open']], 'ks', fillstyle='none', label='AV open')
            plt.plot(results_dict[V][results_dict['n_AV_close']], results_dict[p][results_dict['n_AV_close']], 'k+', fillstyle='none', label='AV close')

       
        elif chamber in ['RV', 'RA']:
            # Extract valve opening and closing timesteps
            n_TV_close = results_dict['n_TV_close']
            n_TV_open = results_dict['n_TV_open']
            n_PV_close = results_dict['n_PV_close']
            n_PV_open = results_dict['n_PV_open']
            n_C_RA = results_dict['n_C_RA']

            # Check if number of transitions timesteps are equal (should be equal to number of cycles in results)
            condition = len(n_TV_close) == len(n_TV_open) == len(n_PV_close) == len(n_PV_open) == len(n_C_RA)
            if not condition:
                warnings.warn(f"Number of right heart cardiac phase timesteps not equal. "
                                f"TV_close: {len(n_TV_close)}, TV_open: {len(n_TV_open)}, "
                                f"PV_close: {len(n_PV_close)}, PV_open: {len(n_PV_open)}, C_RA: {len(n_C_RA)}"
                              )
        
            # Combine valve opening and closing timesteps into single array
            n_close_open = np.sort(np.concatenate((n_TV_close, n_TV_open, n_PV_close, n_PV_open, n_C_RA)))

            # Add last timestep to n_close_open to ensure all timesteps are included
            n_close_open = np.append(n_close_open, -2) # -2 because we add 1 to the index in the loop below

            # Loop over cardiac phase timesteps
            for i in range(len(n_close_open)-1):
                # Reset color cycle after 5 iteration
                if i % 5 == 0:
                    plt.gca().set_prop_cycle(None)

                # Plot cardiac phases of PV loop in different colors
                plt.plot(results_dict[V][n_close_open[i]:n_close_open[i+1]+1], results_dict[p][n_close_open[i]:n_close_open[i+1]+1], '-.')

                if arrows:
                    # Draw an arrow at the midpoint of each cardiac phase
                    n_mid = (n_close_open[i] + n_close_open[i+1]) // 2
                    arrow_direction_x = results_dict[V][n_mid+1] - results_dict[V][n_mid]
                    arrow_direction_y = results_dict[p][n_mid+1] - results_dict[p][n_mid]

                    plt.arrow(results_dict[V][n_mid], results_dict[p][n_mid], 
                            arrow_direction_x, arrow_direction_y, 
                            head_width=2.0, fc='k', ec='k')

            # Plot markers at cardiac phase transitions
            plt.plot(results_dict[V][results_dict['n_TV_open']], results_dict[p][results_dict['n_TV_open']], 'bo', fillstyle='none', label='TV open')
            plt.plot(results_dict[V][results_dict['n_C_RA']], results_dict[p][results_dict['n_C_RA']], 'b*', fillstyle='none', label='Atr. cont.')
            plt.plot(results_dict[V][results_dict['n_TV_close']], results_dict[p][results_dict['n_TV_close']], 'bx', fillstyle='none', label='TV close')
            plt.plot(results_dict[V][results_dict['n_PV_open']], results_dict[p][results_dict['n_PV_open']], 'bs', fillstyle='none', label='PV open')
            plt.plot(results_dict[V][results_dict['n_PV_close']], results_dict[p][results_dict['n_PV_close']], 'b+', fillstyle='none', label='PV close')

        # Reset color cycle to default
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

        # Restore the original font size
        plt.rcParams.update({'font.size': original_font_size})
        
    def plot_results(self, results_dict, output_dir, font_size=12):
        '''
        Plot and save results.

        ARGS:
            - results_dict: Dictionary containing simulation results.
            - output_dir: Directory to save plots.
        '''
        os.makedirs(output_dir, exist_ok=True)

        plt.figure()
        self.plot_pv_loops_without_phase('LV', results_dict, font_size=font_size)
        self.plot_pv_loops_without_phase('RV', results_dict, font_size=font_size)
        self.plot_pv_loops_without_phase('LA', results_dict, font_size=font_size)
        self.plot_pv_loops_without_phase('RA', results_dict, font_size=font_size)
        plt.title('Pressure-Volume Loops')
        plt.xlabel('Volume (mL)')
        plt.ylabel('Pressure (mmHg)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'pv_loops.png'))
        plt.close()

        
        # self.calc_cardiac_phases(results_dict)
        #
        # # Cardiac volumes
        # self.plot_variables_vs_time(results_dict, ['V_LA', 'V_LV', 'V_RA', 'V_RV'],
        #                     'Cardiac Volumes',
        #                     'Volume (mL)',
        #                     output_dir,
        #                     'card_volumes.png',
        #                     phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close',
        #                                     'n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
        #                     font_size=font_size)
        #
        # # Cardiac pressures
        # self.plot_variables_vs_time(results_dict, ['p_LA', 'p_LV', 'p_RA', 'p_RV'],
        #                     'Cardiac Pressures',
        #                     'Pressure (mmHg)',
        #                     output_dir,
        #                     'card_pressures.png',
        #                     phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close',
        #                                     'n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
        #                     font_size=font_size)
        #
        # # Systemic pressures
        # self.plot_variables_vs_time(results_dict, ['p_LA', 'p_LV','p_AR_SYS', 'p_VEN_SYS'],
        #                     'Systemic Pressures',
        #                     'Pressure (mmHg)',
        #                     output_dir,
        #                     'sys_pressures.png',
        #                     phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close'],
        #                     font_size=font_size)
        #
        # # Pulmonary pressures
        # self.plot_variables_vs_time(results_dict, ['p_RA', 'p_RV','p_AR_PUL', 'p_VEN_PUL'],
        #                     'Pulmonary Pressures',
        #                     'Pressure (mmHg)',
        #                     output_dir,
        #                     'pul_pressures.png',
        #                     phase_transitions=['n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
        #                     font_size=font_size)
        #
        # # Systemic flows
        # self.plot_variables_vs_time(results_dict, ['Q_MV', 'Q_AV', 'Q_AR_SYS', 'Q_VEN_SYS'],
        #                     'Systemic Flows',
        #                     'Flow (mL/s)',
        #                     output_dir,
        #                     'sys_flows.png',
        #                     phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close'],
        #                     font_size=font_size)
        #
        # # Pulmonary flows
        # self.plot_variables_vs_time(results_dict, ['Q_TV', 'Q_PV', 'Q_AR_PUL', 'Q_VEN_PUL'],
        #                     'Pulmonary Flows',
        #                     'Flow (mL/s)',
        #                     output_dir,
        #                     'pul_flows.png',
        #                     phase_transitions=['n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
        #                     font_size=font_size)
        #
        # # Cardiac activations
        # self.plot_variables_vs_time(results_dict, ['A_LV', 'A_RV', 'A_LA', 'A_RA'],
        #                     'Cardiac Activations',
        #                     'Activation',
        #                     output_dir,
        #                     'card_activations.png',
        #                     phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close',
        #                                     'n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
        #                     font_size=font_size,
        #                     subplots=True)
        #
        # if self.parameters['chamber_model'] == 'linear':
        #     # Ventricular/atrial elastances
        #     parameters = results_dict['parameters']
        #     results_dict['E_LA'] = parameters['chamber_model_linear']['E_LA_pas'] + parameters['chamber_model_linear']['E_LA_act'] * results_dict['A_LA']
        #     results_dict['E_LV'] = parameters['chamber_model_linear']['E_LV_pas'] + parameters['chamber_model_linear']['E_LV_act'] * results_dict['A_LV']
        #     results_dict['E_RA'] = parameters['chamber_model_linear']['E_RA_pas'] + parameters['chamber_model_linear']['E_RA_act'] * results_dict['A_RA']
        #     results_dict['E_RV'] = parameters['chamber_model_linear']['E_RV_pas'] + parameters['chamber_model_linear']['E_RV_act'] * results_dict['A_RV']
        #
        #     self.plot_variables_vs_time(results_dict, ['E_LA', 'E_LV', 'E_RA', 'E_RV'],
        #                         'Cardiac Elastances',
        #                         'Elastance (mmHg/mL)',
        #                         output_dir,
        #                         'card_elastances.png',
        #                         font_size=font_size)
        #
        # # Plot valve resistances
        # self.plot_variables_vs_time(results_dict, ['R_MV', 'R_AV', 'R_TV', 'R_PV'],
        #                     'Valve Resistances',
        #                     'Resistance (mmHg/mL/s)',
        #                     output_dir,
        #                     'valve_resistances.png',
        #                     subplots=True,
        #                     font_size=font_size)
        #
        # # Plot all volumes
        # self.plot_variables_vs_time(results_dict, ['V_LA', 'V_LV', 'V_RA', 'V_RV', 'V_AR_SYS', 'V_VEN_SYS', 'V_AR_PUL', 'V_VEN_PUL', 'V_tot'],
        #                     'Volumes',
        #                     'Volume (mL)',
        #                     output_dir,
        #                     'all_volumes.png',
        #                     font_size=font_size)

        # Plot ventricular pressure volume loops
        # plt.figure()
        # self.plot_pv_loops_with_phases('LV', results_dict, font_size=font_size)
        # self.plot_pv_loops_with_phases('RV', results_dict, font_size=font_size)
        # plt.title('Ventricular Pressure-Volume Loops')
        # plt.xlabel('Volume (mL)')
        # plt.ylabel('Pressure (mmHg)')
        # plt.legend()
        # plt.savefig(os.path.join(output_dir, 'ventricular_pv_loops.png'))
        # plt.close()

        # self.plot_variables_vs_time(results_dict,
        #                             ['V_LA', 'V_LV', 'V_RA', 'V_RV', 'V_AR_SYS', 'V_VEN_SYS', 'V_AR_PUL', 'V_VEN_PUL',
        #                              'V_tot'],
        #                             'Volumes',
        #                             'Volume (mL)',
        #                             output_dir,
        #                             'all_volumes.png',
        #                             font_size=font_size)
        #
        # # Plot ventricular pressure volume loops
        # plt.figure()
        # self.plot_pv_loops_with_phases('LV', results_dict, font_size=font_size)
        # plt.xlabel('Volume (mL)')
        # plt.ylabel('Pressure (mmHg)')
        # plt.legend()
        # plt.savefig(os.path.join(output_dir, 'lv_pv_loops.png'))
        # plt.close()
        #
        # # Plot atrial pressure volume loops
        # plt.figure()
        # self.plot_pv_loops_with_phases('LA', results_dict, font_size=font_size)
        # self.plot_pv_loops_with_phases('RA', results_dict, font_size=font_size)
        # plt.title('Atrial Pressure-Volume Loops')
        # plt.xlabel('Volume (mL)')
        # plt.ylabel('Pressure (mmHg)')
        # plt.legend()
        # plt.savefig(os.path.join(output_dir, 'atrial_pv_loops.png'))
        # #plt.show()
        # plt.close()

    def plot_metrics(self, sim_metrics, file_path, font_size=12):
        """
        Plot simulation and patient-specific metrics using a bar plot.
        Note: keys in pat_metrics must be a subset of key in sim_metrics.

        Parameters:
        - pat_metrics: dict, patient-specific metrics with labels
        - sim_metrics: dict, metrics
        """

        # Store the original font size
        original_font_size = plt.rcParams['font.size']

        # Increase the font size
        plt.rcParams.update({'font.size': font_size})

        sim_labels = list(sim_metrics.keys())
        width = 0.35  # Width of the bars
        x = np.arange(len(sim_labels))  # Generate x-coordinates for each label

        plt.figure(figsize=(15,3))
        sim_values = np.array([sim_metrics[labels]['Value'] for labels in sim_labels])
        plt.bar(x + width / 2, sim_values, width, label='Simulation', alpha=0.7)
        plt.title('Patient vs Simulation Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        # formatted_labels = [sim_labels[label]['String'] for label in sim_labels]
        plt.xticks(x, sim_labels)  # Set the x-axis labels
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        # Reset the font size
        plt.rcParams.update({'font.size': original_font_size})

    def compute_clinical_metrics(self, results_dict):
        '''
        Compute clinical metrics from simulation results.

        ARGS:
            - results_dict: Dictionary containing simulation results.
        '''

        # Extract relevant simulation metrics
        clinical_metrics = {}

        # Heart rate (beats per minute) and cardiac cycle duration (s)
        self.calc_cardiac_phases(results_dict)
        clinical_metrics['HR'] = {'Value': results_dict['HR'], 'Units': 'BPM'}
        clinical_metrics['T_HB'] = {'Value': results_dict['T_HB'], 'Units': 's'}

        # Aortic blood pressures (mmHg)
        clinical_metrics['P_sys'] = {'Value': np.max(results_dict['p_AR_SYS']),'Units': 'mmHg'}
        clinical_metrics['P_dias'] = {'Value': np.min(results_dict['p_AR_SYS']), 'Units': 'mmHg'}

        # Pulmonary blood pressures (mmHg)
        clinical_metrics['P_sys_pul'] = {'Value': np.max(results_dict['p_AR_PUL']), 'Units': 'mmHg'}
        clinical_metrics['P_dias_pul'] = {'Value': np.min(results_dict['p_AR_PUL']), 'Units': 'mmHg'}

        # Mean systemic arterial pressure (mmHg)
        clinical_metrics['MAP'] = {'Value': np.mean(results_dict['p_AR_SYS']), 'Units': 'mmHg'}

        # Mean pulmonary arterial pressure (mmHg)
        clinical_metrics['mPAP'] = {'Value': np.mean(results_dict['p_AR_PUL']), 'Units': 'mmHg'}

        # Central venous pressure (mean right atrial pressure) (mmHg)
        clinical_metrics['CVP'] = {'Value': np.mean(results_dict['p_RA']), 'Units': 'mmHg'}

        # Pulmonary arterial wedge pressure (mean left atrial pressure) (mmHg)
        clinical_metrics['PAWP'] = {'Value': np.mean(results_dict['p_LA']), 'Units': 'mmHg'}

        # Cardiac min and max volumes (mL)
        clinical_metrics['LVEDV'] = {'Value': np.max(results_dict['V_LV']), 'Units': 'mL'}
        clinical_metrics['LVESV'] = {'Value': np.min(results_dict['V_LV']), 'Units': 'mL'}
        clinical_metrics['RVEDV'] = {'Value': np.max(results_dict['V_RV']), 'Units': 'mL'}
        clinical_metrics['RVESV'] = {'Value': np.min(results_dict['V_RV']), 'Units': 'mL'}
        clinical_metrics['LAEDV'] = {'Value': np.max(results_dict['V_LA']), 'Units': 'mL'}
        clinical_metrics['LAESV'] = {'Value': np.min(results_dict['V_LA']), 'Units': 'mL'}
        clinical_metrics['RAEDV'] = {'Value': np.max(results_dict['V_RA']), 'Units': 'mL'}
        clinical_metrics['RAESV'] = {'Value': np.min(results_dict['V_RA']), 'Units': 'mL'}
  
        # Stroke volumes (mL)
        clinical_metrics['LVSV'] = {'Value': clinical_metrics['LVEDV']['Value'] - clinical_metrics['LVESV']['Value'], 'Units': 'mL'}
        clinical_metrics['RVSV'] = {'Value': clinical_metrics['RVEDV']['Value'] - clinical_metrics['RVESV']['Value'], 'Units': 'mL'}
        clinical_metrics['LASV'] = {'Value': clinical_metrics['LAEDV']['Value'] - clinical_metrics['LAESV']['Value'], 'Units': 'mL'}
        clinical_metrics['RASV'] = {'Value': clinical_metrics['RAEDV']['Value'] - clinical_metrics['RAESV']['Value'], 'Units': 'mL'}
        
        # Ejection fraction (no units)
        clinical_metrics['LVEF'] = {'Value': clinical_metrics['LVSV']['Value'] / clinical_metrics['LVEDV']['Value'], 'Units': '[]'}
        clinical_metrics['RVEF'] = {'Value': clinical_metrics['RVSV']['Value'] / clinical_metrics['RVEDV']['Value'], 'Units': '[]'}

        # Cardiac output (L/min)
        clinical_metrics['CO'] = {'Value': clinical_metrics['HR']['Value'] * clinical_metrics['LVSV']['Value'] / 1000, 'Units': 'L/min'}

        # Systemic and pulmonary vascular resistances (mmHg/(L/min))
        clinical_metrics['SVR'] = {'Value': (clinical_metrics['MAP']['Value'] - clinical_metrics['CVP']['Value']) / (clinical_metrics['CO']['Value']), 'Units': 'mmHg/(L/min)'}
        clinical_metrics['PVR'] = {'Value': (clinical_metrics['mPAP']['Value'] - clinical_metrics['PAWP']['Value']) / (clinical_metrics['CO']['Value']), 'Units': 'mmHg/(L/min)'}

        # Stroke work (mmHg*mL)
        clinical_metrics['LVSW'] = {'Value': -np.trapz(results_dict['p_LV'], results_dict['V_LV']) / results_dict['n_cycles'], 'Units': 'mmHg*mL/cycle'}
        clinical_metrics['RVSW'] = {'Value': -np.trapz(results_dict['p_RV'], results_dict['V_RV']) / results_dict['n_cycles'], 'Units': 'mmHg*mL/cycle'}

        # Save metrics back to results_dict
        results_dict['clinical_metrics'] = clinical_metrics


    def record_output(self, sim_metrics):

        combined_output = np.array([sim_metrics['P_sys']['Value'],
                                    sim_metrics['P_dias']['Value'],
                                    sim_metrics['P_sys_pul']['Value'],
                                    sim_metrics['P_dias_pul']['Value'],
                                    sim_metrics['MAP']['Value'],
                                    sim_metrics['mPAP']['Value'],
                                    sim_metrics['CVP']['Value'],
                                    sim_metrics['PAWP']['Value'],
                                    sim_metrics['LVEDV']['Value'],
                                    sim_metrics['LVESV']['Value'],
                                    sim_metrics['RVEDV']['Value'],
                                    sim_metrics['RVESV']['Value'],
                                    sim_metrics['LAEDV']['Value'],
                                    sim_metrics['LAESV']['Value'],
                                    sim_metrics['RAEDV']['Value'],
                                    sim_metrics['RAESV']['Value'],
                                    sim_metrics['LVSV']['Value'],
                                    sim_metrics['RVSV']['Value'],
                                    sim_metrics['LASV']['Value'],
                                    sim_metrics['RASV']['Value'],
                                    sim_metrics['LVEF']['Value'],
                                    sim_metrics['RVEF']['Value'],
                                    sim_metrics['CO']['Value'],
                                    sim_metrics['SVR']['Value'],
                                    sim_metrics['PVR']['Value'],
                                    sim_metrics['LVSW']['Value'],
                                    sim_metrics['RVSW']['Value'],
                                    self.parameters['BPM'],
                                    self.parameters['PR_interval'],
                                    self.parameters['EMD']
                                   ])

        return combined_output



# Run the simulation if executed as script
if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # General sim parameters
    n_cardiac_cyc = 10
    dt = 0.001  # Here this is a psudo number to determine how many
    save_last_n_cardiac_cycles = 2

    # Run simulation
    sim = Simulation([n_cardiac_cyc, dt, save_last_n_cardiac_cycles])

    sim.get_parameters()
    results_dict = sim.integrate()
