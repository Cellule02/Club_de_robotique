import matplotlib.pyplot as plt
import numpy as np
import random

# Dimensions et paramètres initiaux
width, height = 10, 10
play_margin = 0.5
speed = 0.1
entity_size = 0.4
circling_distance = 1.0

# Position initiale de l'entité alliée (loin du mur)
square1_pos = np.array([width * 0.75, height * 0.5])

# Positions initiales des ennemis (éviter les bords et le mur)
num_enemies = 3
enemy_positions = []
for _ in range(num_enemies):
    x = random.uniform(play_margin * 2, width - play_margin * 2)
    y = random.uniform(play_margin * 2, height - play_margin * 2)
    enemy_positions.append(np.array([x, y]))
enemy_trajectories = [[pos.copy()] for pos in enemy_positions]

# Position du mur et du drapeau
wall_start = np.array([width-0.5, 3])
wall_end = np.array([width-0.5, 7])
flag_pos = np.array([width-0.5, 5])
flag_placed = False

# Initialisation des objectifs
objectives = [
    flag_pos,
    np.array([random.uniform(2, width-2), random.uniform(2, height-2)]),
    np.array([random.uniform(2, width-2), random.uniform(2, height-2)]),
    np.array([random.uniform(2, width-2), random.uniform(2, height-2)])
]
objectives_completed = [False] * len(objectives)

# Liste pour stocker la trajectoire
trajectory_square1 = [square1_pos.copy()]

def clamp_position(pos, width, height, entity_size):
    # Garde l'entité dans les limites de la carte avec marge de sécurité
    return np.array([
        np.clip(pos[0], play_margin, width-play_margin),
        np.clip(pos[1], play_margin, height-play_margin)
    ])

def calculate_acceleration(current_speed, target_speed, max_acceleration=100):
    # Calcule l'accélération entre 0 et 100
    speed_diff = abs(target_speed - current_speed)
    return min(100, speed_diff * 50)  # 50 est un facteur d'échelle

def calculate_path_safety(pos, enemy_positions, target, wall_start, wall_end):
    safety_score = 0
    # Pénalité pour être trop près des bords
    edge_penalty = 0
    if pos[0] < play_margin * 2 or pos[0] > width - play_margin * 2:
        edge_penalty -= 2.0
    if pos[1] < play_margin * 2 or pos[1] > height - play_margin * 2:
        edge_penalty -= 2.0
    safety_score += edge_penalty

    # Analyse des zones dangereuses
    for enemy_pos in enemy_positions:
        dist = np.linalg.norm(enemy_pos - pos)
        safety_score -= 1.5 / (dist + 0.1)
    
    # Bonus pour les chemins éloignés du mur sauf pour le drapeau
    if target is not flag_pos:
        dist_to_wall = abs(pos[1] - wall_start[1])
        if wall_start[0] <= pos[0] <= wall_end[0]:
            safety_score -= 2.0 / (dist_to_wall + 0.1)
    
    # Bonus pour la progression vers l'objectif
    progress = np.dot(target - pos, pos - enemy_positions[0])
    safety_score += 0.5 * progress
    
    return safety_score

def find_best_path(current_pos, target_pos, enemy_positions, wall_start, wall_end):
    best_direction = target_pos - current_pos
    best_score = float('-inf')
    angles = np.linspace(-np.pi, np.pi, 16)
    
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        test_pos = current_pos + direction * speed * 5
        score = calculate_path_safety(test_pos, enemy_positions, target_pos, wall_start, wall_end)
        if score > best_score:
            best_score = score
            best_direction = direction
    
    return best_direction

def process_ally_movement(current_pos, objectives, enemy_positions, wall_start, wall_end, speed):
    # Paramètres physiques
    MAX_SPEED = speed * 5.0
    MIN_SPEED = speed * 0.2
    BASE_ACCEL = 0.02
    MAX_ACCEL = 0.08
    LOOK_AHEAD_STEPS = 50

    # Système de points corrigé
    POINTS = {
        'FLAG': {
            'base_score': 100,           # Score de base pour le drapeau
            'enemy_near': -50,           # Pénalité ennemie proche
            'enemy_far': -20,            # Pénalité ennemie loin
            'wall_follow': -30,           # Bonus suivi de mur
            'progress': 40               # Bonus progression
        },
        'OBJECTIVE': {
            'base_score': 150,            # Score de base pour objectif
            'enemy_near': -80,           # Pénalité ennemie plus forte
            'enemy_far': 30,            # Pénalité moyenne distance
            'wall_avoid': 20,            # Bonus évitement mur
            'progress': 30               # Bonus progression
        }
    }

    def evaluate_position(pos, target, is_flag=False):
        points = 0
        point_type = POINTS['FLAG'] if is_flag else POINTS['OBJECTIVE']
        
        # Distance à l'objectif et aux ennemis
        dist_to_target = np.linalg.norm(target - pos)
        nearest_enemy_dist = min(np.linalg.norm(enemy_pos - pos) for enemy_pos in enemy_positions)
        
        # Si l'ennemi est plus loin que l'objectif, focus sur l'objectif
        if nearest_enemy_dist > dist_to_target + 1.0:
            points += point_type['base_score'] * 2.0 / (1 + dist_to_target)
            points += point_type['progress'] * 1.5
        else:
            # Sinon, équilibrer entre objectif et évitement
            points += point_type['base_score'] / (1 + dist_to_target)
            for enemy_pos in enemy_positions:
                dist_to_enemy = np.linalg.norm(enemy_pos - pos)
                if dist_to_enemy < 2.0:
                    points += point_type['enemy_near'] * (2.0 - dist_to_enemy) * 0.5
                elif dist_to_enemy < 4.0:
                    points += point_type['enemy_far'] * (4.0 - dist_to_enemy) * 0.5

        # Bonus mur pour le drapeau
        if is_flag:
            dist_to_wall = abs(pos[0] - wall_start[0])
            if dist_to_wall < 1.5:
                points += point_type['wall_follow'] * (1.5 - dist_to_wall)
        
        return points

    # État du mouvement
    if not hasattr(process_ally_movement, 'state'):
        process_ally_movement.state = {
            'velocity': np.zeros(2),
            'acceleration': np.zeros(2),
            'last_direction': np.zeros(2),
            'sprint_cooldown': 0,
            'planned_path': []
        }
    
    state = process_ally_movement.state
    
    # Planification du chemin à long terme
    def plan_path(start_pos, target):
        best_path = []
        best_score = float('-inf')
        is_flag = np.array_equal(target, objectives[0])
        
        for angle in np.linspace(-np.pi, np.pi, 16):
            current_path = [start_pos]
            total_score = 0
            test_pos = start_pos.copy()
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            for step in range(LOOK_AHEAD_STEPS):
                next_pos = test_pos + direction * speed * (step + 1)
                pos_score = evaluate_position(next_pos, target, is_flag)
                total_score += pos_score / (step + 1)
                
                if np.linalg.norm(next_pos - target) < np.linalg.norm(test_pos - target):
                    total_score += POINTS['FLAG' if is_flag else 'OBJECTIVE']['progress']
                
                current_path.append(next_pos)
                test_pos = next_pos
            
            if total_score > best_score:
                best_score = total_score
                best_path = current_path
        
        return best_path, best_score
    
    # Mise à jour du chemin planifié si nécessaire
    if len(state['planned_path']) < 2:
        target = objectives[0] if not objectives_completed[0] else next(obj for i, obj in enumerate(objectives) if not objectives_completed[i])
        state['planned_path'], _ = plan_path(current_pos, target)
    
    # Utilisation du chemin planifié pour la direction
    next_waypoint = state['planned_path'][1] if len(state['planned_path']) > 1 else target
    desired_direction = next_waypoint - current_pos
    if np.linalg.norm(desired_direction) > 0:
        desired_direction = desired_direction / np.linalg.norm(desired_direction)
    
    # Mise à jour du chemin planifié
    if np.linalg.norm(current_pos - state['planned_path'][0]) > speed:
        state['planned_path'].pop(0)

    # Évaluation des menaces
    threats = []
    for enemy_pos in enemy_positions:
        dist = np.linalg.norm(enemy_pos - current_pos)
        if dist < 3.0:
            threats.append((enemy_pos, dist))
    
    # Décision de direction
    if threats:
        # Mode évitement
        escape_dir = sum([(current_pos - pos) / (d * d) for pos, d in threats], np.zeros(2))
        if np.linalg.norm(escape_dir) > 0:
            escape_dir = escape_dir / np.linalg.norm(escape_dir)
            desired_direction = escape_dir * 0.7 + (next_waypoint - current_pos) * 0.3
    else:
        # Mode poursuite
        desired_direction = next_waypoint - current_pos
    
    if np.linalg.norm(desired_direction) > 0:
        desired_direction = desired_direction / np.linalg.norm(desired_direction)
    
    # Calcul de l'accélération
    if threats:
        target_speed = MAX_SPEED
        accel_strength = MAX_ACCEL
        state['sprint_cooldown'] = 10
    elif np.linalg.norm(next_waypoint - current_pos) < 1.0:
        target_speed = MIN_SPEED
        accel_strength = BASE_ACCEL
    else:
        target_speed = speed * 2.0
        accel_strength = BASE_ACCEL
        if state['sprint_cooldown'] > 0:
            state['sprint_cooldown'] -= 1
    
    # Application de l'accélération avec inertie
    desired_velocity = desired_direction * target_speed
    acceleration = (desired_velocity - state['velocity']) * accel_strength
    
    # Limitation de l'accélération
    accel_magnitude = np.linalg.norm(acceleration)
    if accel_magnitude > MAX_ACCEL:
        acceleration = acceleration * (MAX_ACCEL / accel_magnitude)
    
    # Mise à jour de la vitesse
    state['velocity'] += acceleration
    
    # Limitation de la vitesse
    speed_magnitude = np.linalg.norm(state['velocity'])
    if speed_magnitude > MAX_SPEED:
        state['velocity'] *= MAX_SPEED / speed_magnitude
    
    # Calcul du mouvement final
    movement = state['velocity']
    state['last_direction'] = desired_direction
    
    # Position finale avec contraintes
    new_pos = current_pos + movement
    new_pos = clamp_position(new_pos, width, height, entity_size)
    final_movement = new_pos - current_pos
    
    # Retour d'informations
    dx, dy = final_movement[0], final_movement[1]
    current_speed = np.linalg.norm(state['velocity'])
    
    print(f"dx={dx:.3f}, dy={dy:.3f}, vitesse={current_speed:.2f}")
    return final_movement, {
        'dx': dx,
        'dy': dy,
        'speed': current_speed
    }

def update_enemy_movement(current_pos, ally_pos, wall_start, wall_end, speed):
    # Paramètres
    ATTACK_DISTANCE = 4.0
    WALL_BUFFER = 2.0
    MAX_SPEED = speed * 1.5
    
    # Direction vers l'allié
    direction = ally_pos - current_pos
    distance = np.linalg.norm(direction)
    if distance > 0:
        direction = direction / distance
    
    # Calcul de la vitesse selon la distance
    current_speed = speed
    if distance < ATTACK_DISTANCE:
        current_speed = MAX_SPEED
    
    # Évitement du mur
    wall_avoid = np.zeros(2)
    if wall_start[1] <= current_pos[1] <= wall_end[1]:
        dist_to_wall = abs(current_pos[0] - wall_start[0])
        if dist_to_wall < WALL_BUFFER:
            wall_avoid[0] = np.sign(current_pos[0] - wall_start[0]) * (WALL_BUFFER - dist_to_wall)
    
    # Mouvement final
    movement = (direction + wall_avoid * 0.5) * current_speed
    new_pos = current_pos + movement
    new_pos = clamp_position(new_pos, width, height, entity_size)
    
    return new_pos

# Modifier la boucle principale pour utiliser la nouvelle fonction de mise à jour des ennemis
movements = []
while not all(objectives_completed):
    movement, traj_info = process_ally_movement(square1_pos, objectives, 
                                              enemy_positions, wall_start, wall_end, speed)
    square1_pos += movement
    movements.append(movement)
    
    # Mettre à jour les positions des ennemis
    for j in range(num_enemies):
        enemy_positions[j] = update_enemy_movement(enemy_positions[j], square1_pos,
                                                 wall_start, wall_end, speed)
        enemy_trajectories[j].append(enemy_positions[j].copy())

    trajectory_square1.append(square1_pos.copy())

    # Vérifier la complétion des objectifs
    for i, objective in enumerate(objectives):
        if not objectives_completed[i] and np.linalg.norm(square1_pos - objective) < 0.2:
            objectives_completed[i] = True
            if i == 0:  # Si c'est le drapeau
                flag_placed = True
                print("Drapeau placé!")
            print(f"Objectif {i} complété!")
            print(f"Mouvement: dx={movement[0]:.3f}, dy={movement[1]:.3f}")

    # Vérifier les collisions avec les ennemis
    for enemy_pos in enemy_positions:
        if np.linalg.norm(square1_pos - enemy_pos) < entity_size:
            print("Collision avec un ennemi! Fin de la simulation.")
            plt.close()
            exit()

    # Visualisation
    plt.clf()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-1, width+1)
    plt.ylim(-1, height+1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Simulation de navigation')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Dessiner le mur
    plt.plot([wall_start[0], wall_end[0]], [wall_start[1], wall_end[1]], 'k-', linewidth=5, label='Mur')

    # Dessiner le drapeau
    if not flag_placed:
        plt.plot(flag_pos[0], flag_pos[1], 'k^', markersize=15, label='Drapeau')
    else:
        plt.plot(flag_pos[0], flag_pos[1], 'g^', markersize=15, label='Drapeau (placé)')

    # Dessiner l'allié
    ally_patch = plt.Rectangle(square1_pos - entity_size/2, entity_size, entity_size, 
                             color='blue', alpha=0.8, label='Allié')
    plt.gca().add_patch(ally_patch)

    # Dessiner les ennemis avec des triangles rouges
    for enemy_pos in enemy_positions:
        plt.plot(enemy_pos[0], enemy_pos[1], 'rv', markersize=15, alpha=0.8)

    # Dessiner les objectifs avec des cercles plus visibles
    for i, objective in enumerate(objectives):
        color = 'green' if objectives_completed[i] else 'orange'
        circle = plt.Circle(objective, 0.3, color=color, alpha=0.6)
        plt.gca().add_patch(circle)
        plt.text(objective[0], objective[1], f'Obj {i+1}', 
                horizontalalignment='center', verticalalignment='bottom')

    # Dessiner les trajectoires en pointillés plus fins
    plt.plot([p[0] for p in trajectory_square1], [p[1] for p in trajectory_square1], 
             'b--', alpha=0.3, linewidth=1)
    for trajectory in enemy_trajectories:
        plt.plot([p[0] for p in trajectory], [p[1] for p in trajectory], 
                 'r--', alpha=0.3, linewidth=1)

    plt.legend(loc='upper right')
    plt.pause(0.1)

print("Tous les objectifs sont atteints!")
print(f"Nombre total de mouvements: {len(movements)}")
plt.show()
