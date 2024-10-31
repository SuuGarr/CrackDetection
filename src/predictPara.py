from scipy.optimize import minimize

EPSILON = 1e-6

# Objective functions
def objective_1(x, original_current, original_speed):
    current, speed = x
    return abs((((10 * original_current + 0.04 * original_current**2) / original_speed) +
                ((((10 + 0.04 * current) / speed) * (current - original_current)) +
                ((((-0.04 * current**2) - (10 * current)) / speed**2) * (speed - original_speed)))) - 180)

def objective_2(x, original_current, original_speed):
    current, speed = x
    return abs((((10 * original_current + 0.04 * original_current**2) / original_speed) +
                ((((10 + 0.04 * current) / speed) * (current - original_current)) +
                ((((-0.04 * current**2) - (10 * current)) / speed**2) * (speed - original_speed)))) - 75)

def objective_3(x, original_current, original_speed):
    current, speed = x
    return abs((((10 * original_current + 0.04 * original_current**2) / original_speed) +
                ((((10 + 0.04 * current) / speed) * (current - original_current)) +
                ((((-0.04 * current**2) - (10 * current)) / speed**2) * (speed - original_speed)))) - 53)

# Constraints
def constraint1(x):
    current, speed = x
    if speed < EPSILON:
        return -1  # Indicate a failed constraint if speed is zero or below
    return 180 - (10 * current + 0.04 * current**2) / speed

def constraint2(x):
    current, speed = x
    if speed < EPSILON:
        return -1
    return (10 * current + 0.04 * current**2) / speed - 75

def constraint3(x):
    current, speed = x
    return -0.05446698341575029 * current - 0.1818879257122603 * speed + 4.813862370750962

def predict_parameters(initial_current, initial_speed):
    initial_speed = max(initial_speed, EPSILON)
    x0 = [initial_current, initial_speed]

    bounds = [(0, None), (EPSILON, None)]

    constraints = ({'type': 'ineq', 'fun': constraint1},
                   {'type': 'ineq', 'fun': constraint2},
                   {'type': 'ineq', 'fun': constraint3})

    if constraint1(x0) >= 0 and constraint2(x0) >= 0 and constraint3(x0) >= 0:
        return initial_current, initial_speed
    else:
        if constraint1(x0) < 0:
            result = minimize(objective_1, x0, args=(initial_current, initial_speed), method='COBYLA',
                              bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        elif constraint2(x0) < 0:
            result = minimize(objective_2, x0, args=(initial_current, initial_speed), method='COBYLA',
                              bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        elif constraint3(x0) < 0:
            result = minimize(objective_3, x0, args=(initial_current, initial_speed), method='COBYLA',
                              bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        else:
            raise ValueError('Invalid Input')

        optimized_current = round(result.x[0])
        optimized_speed = result.x[1]

        return optimized_current, optimized_speed
