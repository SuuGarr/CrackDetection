from scipy.optimize import minimize
import math


def objective_1(x):
    current, speed = x
    original_current, original_speed = x0
    return abs((((10*original_current + 0.04*original_current**2) / original_speed)+ ((((10 + 0.04*current)/speed)*(current-original_current))+((((-0.04*current**2 )- (10*current )) / speed**2)*(speed-original_speed))))-180)

def objective_2(x):
    current, speed = x
    original_current, original_speed = x0
    return abs((((10*original_current + 0.04*original_current**2) / original_speed)+ ((((10 + 0.04*current)/speed)*(current-original_current))+((((-0.04*current**2 )- (10*current )) / speed**2)*(speed-original_speed))))-75)

def objective_3(x):
    current, speed = x
    original_current, original_speed = x0
    return abs((((10*original_current + 0.04*original_current**2) / original_speed)+ ((((10 + 0.04*current)/speed)*(current-original_current))+((((-0.04*current**2 )- (10*current )) / speed**2)*(speed-original_speed))))-53)

def constraint1(x):
    current, speed = x
    return 180 - (10*current + 0.04*current**2) / speed

def constraint2(x):
    current, speed = x
    return (10*current + 0.04*current**2) / speed - 75

def constraint3(x):
    current, speed = x
    return -0.05446698341575029*current - 0.1818879257122603*speed + 4.813862370750962


I = float(input('Current: '))
v = float(input('Speed: '))
x0 = [I, v]

bounds = [(0, None), (0, None)] 
 
# Define constraints
constraints = ({'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2},
               {'type': 'ineq', 'fun': constraint3})

if constraint1(x0) >= 0 and constraint2(x0) >= 0 and constraint3(x0) >= 0:
    result = x0
    print("Optimized Current:", result[0])
    print("Optimized Speed:", result[1])
else:
    # Minimize corresponding objective function based on which constraint is violated
    if constraint1(x0) < 0:
        result = minimize(objective_1, x0, method='COBYLA', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    elif constraint2(x0) < 0:
        result = minimize(objective_2, x0, method='COBYLA', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    elif constraint3(x0) < 0:
        result = minimize(objective_3, x0, method='COBYLA', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
    else:
        print('Invalid Input')

    # Round the optimized values to integers
    optimized_current = result.x[0]
    optimized_speed = result.x[1]

    print("Optimized Current:", optimized_current)
    print("Optimized Speed:", optimized_speed)