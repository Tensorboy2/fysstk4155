import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

'''
Module for generating the harmonic oscillator data.
'''

g = 9.81  
L = 1.0  

theta_0 = np.pi / 8
omega_0 = 0.0       

t_max = 4 
t_span = (0, t_max)  
t_eval = np.linspace(0, t_max, 251)  

def pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

y0 = [theta_0, omega_0]

solution = solve_ivp(pendulum, t_span, y0, t_eval=t_eval, method='RK45')

time_data = solution.t
theta_data = solution.y[0]

df = pd.DataFrame({
    'time': time_data,
    'angle': theta_data
})
path = '/home/sigvar/1_semester/fysstk4155/fysstk_2/project_3/src/data/harmonic_occilator_data/'
df.to_csv(path+'h_o_data.csv')

print(f'Data stored as h_o_data.csv in dir: project_3/src/data/harmonic_occilator_data')
print(df.head())

