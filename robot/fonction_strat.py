import numpy as np

# Position actuelle du robot
pos = np.array([7.5, 5.0])

# Liste des objectifs
objectives = [
    np.array([9.5, 5.0]),   # Objectif 1 (ex: drapeau)
    np.array([3.0, 8.0]),   # Objectif 2
    np.array([2.0, 2.0])    # Objectif 3
]

# Objectifs atteints ou non
objectives_completed = [False, False, True]

# Positions des ennemis
enemy_positions = [
    np.array([6.0, 5.5]),
    np.array([8.0, 6.0])
]

# Coordonnées du mur
wall_start = np.array([9.5, 3.0])
wall_end = np.array([9.5, 7.0])

# Définition des dimensions du terrain et de la marge de sécurité
play_margin = 0.5  # marge de sécurité pour éviter les bords
width = 10.0       # largeur du terrain
height = 10.0      # hauteur du terrain

def score_direction(pos, direction, objectives, objectives_completed, enemy_positions, wall_start, wall_end, params):
    """
    Calcule un score pour une direction donnée.
    params: dictionnaire de pondérations/scoring
    """
    score = 0
    next_pos = pos + direction * params['step_size']

    # Score distance à l'objectif le plus proche non atteint
    for i, obj in enumerate(objectives):
        if not objectives_completed[i]:
            dist = np.linalg.norm(next_pos - obj)
            score += params['objective_weight'] / (dist + 0.1)
            # Bonus si très proche
            if dist < 0.5:
                score += params['objective_bonus']

    # Pénalité proximité ennemis
    for enemy_pos in enemy_positions:
        dist = np.linalg.norm(next_pos - enemy_pos)
        if dist < params['enemy_avoid_radius']:
            score -= params['enemy_penalty'] * (params['enemy_avoid_radius'] - dist)

    # Pénalité proximité mur
    if wall_start[1] <= next_pos[1] <= wall_end[1]:
        dist_to_wall = abs(next_pos[0] - wall_start[0])
        if dist_to_wall < params['wall_buffer']:
            score -= params['wall_penalty'] * (params['wall_buffer'] - dist_to_wall)

    # Pénalité bords de carte
    if next_pos[0] < play_margin or next_pos[0] > width - play_margin or \
       next_pos[1] < play_margin or next_pos[1] > height - play_margin:
        score -= params['edge_penalty']

    return score

def decide_next_move(pos, objectives, objectives_completed, enemy_positions, wall_start, wall_end):
    # Paramètres de scoring ajustables
    params = {
        'step_size': 0.5,
        'objective_weight': 30,
        'objective_bonus': 20,
        'enemy_avoid_radius': 2.0,
        'enemy_penalty': 40,
        'wall_buffer': 1.0,
        'wall_penalty': 30,
        'edge_penalty': 100
    }

    best_score = float('-inf')
    best_direction = np.zeros(2)
    best_orientation = 0.0

    # Tester plusieurs directions (par exemple, 16 autour du robot)
    for angle in np.linspace(-np.pi, np.pi, 16):
        direction = np.array([np.cos(angle), np.sin(angle)])
        score = score_direction(pos, direction, objectives, objectives_completed, enemy_positions, wall_start, wall_end, params)
        if score > best_score:
            best_score = score
            best_direction = direction
            best_orientation = angle

    # Si le score est trop bas, on peut décider de ne pas bouger (risque trop élevé)
    if best_score < 0:
        print("Aucune direction sûre, le robot attend.")
        return np.zeros(2), best_orientation, 0

    # Sinon, avancer dans la meilleure direction
    move = best_direction * params['step_size']
    distance = np.linalg.norm(move)
    return move, best_orientation, distance

# Exemple d'utilisation dans ta boucle principale :
# move, orientation, distance = decide_next_move(pos, objectives, objectives_completed, enemy_positions, wall_start, wall_end)
# Appel de la fonction
move, orientation, distance = decide_next_move(
    pos, objectives, objectives_completed, enemy_positions, wall_start, wall_end
)

print("Déplacement suggéré :", move)
print("Orientation (radian) :", orientation)
print("Distance à parcourir :", distance)
