import numpy as np
from scipy.integrate import odeint, RK45
import matplotlib.pyplot as plt

from integrators import rk4, euler_newton

class aeroelastic_spring_mdl:
    def __init__(self, f_flap, A_flap, cpy, G_section, J_section, g, I, k_spring=1.0):
        self.f = f_flap
        self.A = A_flap
        #self.k_spring = k_spring #An equivalent torsional spring coefficient (Could be calculated with length to center of pressure in spanwise direction)
        self.k_spring = 0.05*(G_section * J_section / cpy)
        self.g = g
        self.Ixx = I[0][0]
        self.Ixy = I[0][1]
        self.f_crit = (np.sqrt(self.k_spring/self.Ixx))/(2*np.pi) #Critical frequency in Hz

        # I is a 2 x 2 inertia matrix (nd array) [[Ixx, Ixy], [Ixy, Iyy]] with respect to the wing root location at leading edge -> Can be calculated from membrane specific mass

    def ode_definition(self,x,t,M_aero_root):
        # x is the state vecotr, corresponding to x = [psi, psi_dot]
        # t is the current time
        # M_aero_root comes from Ptera, sum of all aerodynamic Moments about the leading edge twist direction that are computed from segements in Ptera 
        phi = self.A * np.sin(2*np.pi*self.f*t)
        phi_dot = 2*np.pi*self.f*self.A * np.cos(2*np.pi*self.f*t)
        phi_dot_dot = -((2*np.pi*self.f*self.A)**2) * np.sin(2*np.pi*self.f*t)

        psi, psi_dot = x
        x0_dot = psi_dot
        x1_dot = (1/self.Ixx)*((M_aero_root-self.k_spring*psi) + self.Ixy*phi_dot_dot*np.cos(psi) + 0.5*self.Ixx*(phi_dot**2)*np.sin(2*psi)) #Mass effects are neglected in this equation (mass is low comapred to flapping, reasonable assumption)
        print(self.k_spring*psi)
        dxdt = [x0_dot, x1_dot]

        return dxdt
    
    def state_update_call(self,x_curr,M_aero_root_curr,dt,t):
        # Run single step loop with updated Aerodynamic moments to give updated x = [psi, psi_dot]
        x = rk4(self.ode_definition, x_curr, t, dt, M_aero_root_curr)

        return x
    
if __name__ == "__main__":
    f = 5 #Hz
    A_flap = 30*(np.pi/180) #radians maximum amplitude
    b_span = 0.3 #meters
    cpy = b_span/2
    t_spar = 0.01 # 1 cm width of leading edge spar to contribute to stiffness
    J_section = np.pi/2 * (t_spar**4) #Assuming circular cross-section of leading edge spar
    G_spar = 5e9 #Shear modulus of a typical carbon fibre prepreg in Pa
    g = 9.81
    I = np.array([[1e-2, 1e-1], [1e-1, 2e-1]]) # calculate from first principles using integral formulation + Beam inertia. Specific mass is membrane density * membrane thickness

    # Initialize model
    mdl = aeroelastic_spring_mdl(f, A_flap, cpy, G_spar, J_section, g, I)

    psi_0 = 0.0
    psi_dot_0 = 0.0 
    x_0 = [psi_0, psi_dot_0]
    dt_sim = 1e-3
    N_sim = 1000
    t_sim = N_sim*dt_sim
    t = np.linspace(0, t_sim, N_sim + 1)
    M_aero_root_const = 0.0 #Aero currently turned off

    # # Simulate ode without aero
    x = odeint(mdl.ode_definition, x_0, t, args=(M_aero_root_const,))
    plt.plot(t, x[:, 0]*(180/np.pi), 'b', label='psi(t) in degrees')
    plt.plot(t, A_flap * np.sin(2*np.pi*f*t)*(180/np.pi))
    #plt.plot(t, x[:, 1], 'g', label='psi_dot(t)')
    plt.show()

    print("Approximated critical excitation resonance frequency: " + str(mdl.f_crit))

    # Solve simulation with changing aero one step at a time in loop
    # x_sol = [x_0]
    # for k in t[:-1]:
    #     M_aero = 0.0 #-5.0*np.sin(2*np.pi*f*k)
    #     x_sol.append(mdl.state_update_call(x_sol[-1],M_aero,dt_sim,k))
    
    # x_sol = np.array(x_sol)
    # plt.plot(t, x_sol[:,0]*(180/np.pi))




    





        