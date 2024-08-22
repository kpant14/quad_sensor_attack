import picos
import numpy as np

def solve_lmi(A, B, C, W, Bo, Lo, alpha=0, verbosity=0):
    rho = 0.7
    d_max = 1
    
    n = A.shape[0]
    prob = picos.Problem()
    
    G1 = picos.RealVariable('G1', (n,n))
    G2 = picos.RealVariable('G2', (n,n))
    G12 = picos.RealVariable('G12', (n,n))

    Q1 = picos.SymmetricVariable('Q1', (n,n))
    Q2 = picos.SymmetricVariable('Q2', (n,n))
    
    Y = picos.RealVariable('Y', (2, n))
    Z = picos.RealVariable('Z', (n, n))
    
    gamma = picos.RealVariable('gamma')

    Q_curly = np.block([[np.eye(n),np.zeros((n,n))],[np.zeros((n,n)), np.eye(n)]])
    T = np.block([[np.eye(n), np.zeros((n,n)),np.zeros((n,n)),-np.eye(n)],
             [np.eye(n), np.zeros((n,n)),np.zeros((n,n)),-np.eye(n)]])
    Q_curly_hat = Q_curly @ T 
    Q = picos.block([[Q1 , np.zeros((n,n))],[np.zeros((n,n)), Q2]])
    G = picos.block([[G1 , np.zeros((n,n))],[G12, G2]])
    G_tilde = picos.block([[G1 + G1.T - Q1, G12.T],[G12, G2 + G2.T - Q2]])
    spade_1 = picos.block([[Z + Bo@Y + Lo@C@(G12+G1), Lo@C@G2],[(A - Lo@C)@(G12+G1) - Z + (B - Bo)@Y, (A - Lo@C)@G2]])
    spade_2 = picos.block([[np.zeros((n,n)), -Lo],[W, -Lo]])
    block_eq1 = picos.block([
        [(1-rho)*G_tilde, np.zeros((2*n,2*n)),spade_1.T ],
        [np.zeros((2*n,2*n)), (rho/d_max)*np.eye(2*n), spade_2.T, np.zeros((2*n,2*n))],
        [spade_1, spade_2, Q,  np.zeros((2*n,2*n))],
        [Q_curly_hat*G, np.zeros((2*n,2*n)),np.zeros((2*n,2*n)), gamma*np.eye(2*n)]])
    
    prob.add_constraint(block_eq1 >> 0) 
    #prob.add_constraint(Y >> 1)
    #prob.add_constraint(Z >> 1)
    #prob.add_constraint(gamma >> 0.6)
    prob.set_objective('min', gamma)
    prob.options["*_tol"] = 1e-6
    try:
        prob.solve(options={'verbosity': verbosity})
        cost = gamma.value
    except Exception as e:
        print(e)
        cost = -1
    return {
        'cost': cost,
        'prob': prob,
        'A': np.round(np.array(Y.value), n),
        'F': np.round(np.array(Z.value @ np.linalg.inv(Y.value)), n),
        'gamma': gamma
    }