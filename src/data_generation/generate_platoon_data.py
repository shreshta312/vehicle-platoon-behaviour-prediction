import pandas as pd
import numpy as np
import random
import warnings

# --- 1. Simulation Constants ---
NUM_CARS = 60
NUM_LANES = 3
ROAD_LENGTH = 1500.0
SIMULATION_STEPS = 75000
DT = 0.1
MAX_SENSOR_DIST = 150.0
SAFE_TIME_GAP = 1.5
OVERTAKE_CLEAR_DIST_REAR = 30.0
OVERTAKE_CLEAR_DIST_FRONT = 50.0

# --- 2. Target "y" Labels ---
DECISION_BRAKE = 0
DECISION_FOLLOW = 1
DECISION_LANE_LEFT = 2
DECISION_LANE_RIGHT = 3

# --- 3. The "World" Simulation ---
cars = []
for i in range(NUM_CARS):
    cars.append({
        'id': i,
        'lane': random.randint(0, NUM_LANES - 1),
        'pos': random.uniform(0, ROAD_LENGTH),
        'vel': random.uniform(15.0, 30.0),
        'desired_vel': random.uniform(25.0, 35.0)
    })

def find_neighbors(ego_car, all_cars):
    neighbors = {
        'front_dist': MAX_SENSOR_DIST, 'front_vel': 0.0,
        'left_front_dist': MAX_SENSOR_DIST, 'left_rear_dist': MAX_SENSOR_DIST,
        'right_front_dist': MAX_SENSOR_DIST, 'right_rear_dist': MAX_SENSOR_DIST,
    }

    for other_car in all_cars:
        if ego_car['id'] == other_car['id']: continue
        dist = other_car['pos'] - ego_car['pos']
        if dist < -ROAD_LENGTH / 2: dist += ROAD_LENGTH
        if dist > ROAD_LENGTH / 2: dist -= ROAD_LENGTH

        if other_car['lane'] == ego_car['lane'] and dist > 0:
            if dist < neighbors['front_dist']:
                neighbors['front_dist'] = dist
                neighbors['front_vel'] = other_car['vel']
        elif other_car['lane'] == ego_car['lane'] - 1:
            if dist > 0:
                if dist < neighbors['left_front_dist']: neighbors['left_front_dist'] = dist
            else:
                if abs(dist) < neighbors['left_rear_dist']: neighbors['left_rear_dist'] = abs(dist)
        elif other_car['lane'] == ego_car['lane'] + 1:
            if dist > 0:
                if dist < neighbors['right_front_dist']: neighbors['right_front_dist'] = dist
            else:
                if abs(dist) < neighbors['right_rear_dist']: neighbors['right_rear_dist'] = abs(dist)
    return neighbors

# --- 4. The "Expert" AI Logic ---
def get_expert_decision(ego_car, neighbors):
    time_gap = MAX_SENSOR_DIST
    if ego_car['vel'] > 0:
        time_gap = neighbors['front_dist'] / ego_car['vel']
    
    if time_gap < SAFE_TIME_GAP:
        return DECISION_BRAKE

    is_stuck = (
        neighbors['front_dist'] < (MAX_SENSOR_DIST * 0.9) and
        ego_car['vel'] < ego_car['desired_vel'] and
        neighbors['front_vel'] < ego_car['vel']
    )
    if is_stuck:
        can_go_left = (
            ego_car['lane'] > 0 and
            neighbors['left_rear_dist'] > OVERTAKE_CLEAR_DIST_REAR and
            neighbors['left_front_dist'] > OVERTAKE_CLEAR_DIST_FRONT
        )
        if can_go_left: return DECISION_LANE_LEFT
        can_go_right = (
            ego_car['lane'] < (NUM_LANES - 1) and
            neighbors['right_rear_dist'] > OVERTAKE_CLEAR_DIST_REAR and
            neighbors['right_front_dist'] > OVERTAKE_CLEAR_DIST_FRONT
        )
        if can_go_right: return DECISION_LANE_RIGHT

    return DECISION_FOLLOW

def update_simulation(all_cars):
    for car in all_cars:
        neighbors = find_neighbors(car, all_cars)
        if neighbors['front_dist'] < (car['vel'] * SAFE_TIME_GAP * 1.2):
            car['vel'] *= 0.95
        elif car['vel'] < car['desired_vel']:
            car['vel'] *= 1.05
        car['vel'] = max(5.0, min(car['vel'], 40.0))
        car['pos'] += car['vel'] * DT
        if car['pos'] > ROAD_LENGTH: car['pos'] -= ROAD_LENGTH
        if car['pos'] < 0: car['pos'] += ROAD_LENGTH
        if random.random() < 0.001:
            car['lane'] = (car['lane'] + random.choice([-1, 1]))
            car['lane'] = max(0, min(car['lane'], NUM_LANES - 1))

# --- 5. Main Data Generation Loop ---
print("Generating new, smarter dataset...")
data = []
warnings.filterwarnings('ignore', category=RuntimeWarning)

for step in range(SIMULATION_STEPS):
    ego_car = random.choice(cars)
    neighbors = find_neighbors(ego_car, cars)
    decision = get_expert_decision(ego_car, neighbors)

    front_rel_vel = 0.0
    if neighbors['front_dist'] < MAX_SENSOR_DIST:
        front_rel_vel = neighbors['front_vel'] - ego_car['vel']
    
    time_gap = MAX_SENSOR_DIST
    if ego_car['vel'] > 0.1:
        time_gap = neighbors['front_dist'] / ego_car['vel']
    
    features = [
        ego_car['vel'],
        neighbors['front_dist'],
        front_rel_vel,
        time_gap, # Our new, "magic" feature
        neighbors['left_front_dist'],
        neighbors['left_rear_dist'],
        neighbors['right_front_dist'],
        neighbors['right_rear_dist']
    ]
    data.append(features + [decision])

    if step % 10 == 0:
        update_simulation(cars)

# --- 6. Create and Save DataFrame ---
print("Generation complete. Creating DataFrame...")
feature_names = [
    'ego_vel', 'front_dist', 'front_rel_vel', 'time_gap',
    'left_front_dist', 'left_rear_dist',
    'right_front_dist', 'right_rear_dist'
]
target_name = 'decision'

df = pd.DataFrame(data, columns=feature_names + [target_name])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df.to_csv('platoon_data.csv', index=False)
print(f"Successfully saved 'platoon_data.csv' with {len(df)} samples.")