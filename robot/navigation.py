import numpy as np
import time
from math import pi, cos, sin, atan2

class RaspblockNavigator:
    def __init__(self):
        # Constantes physiques du Raspblock
        self.WHEEL_RADIUS = 0.0325  # Rayon des roues en mètres
        self.WHEEL_BASE = 0.21     # Distance entre les roues en mètres
        self.MAX_SPEED = 0.5        # Vitesse maximale en m/s
        self.MIN_SPEED = 0.1        # Vitesse minimale en m/s
        
        # État actuel
        self.current_pos = np.array([0.0, 0.0])  # Position (x, y)
        self.current_orientation = 0.0            # Orientation en radians
        
        # PID pour le contrôle de direction
        self.Kp_angle = 2.0
        self.Ki_angle = 0.0
        self.Kd_angle = 0.5
        self.angle_integral = 0.0
        self.last_angle_error = 0.0
        
        # PID pour le contrôle de rotation Z
        self.Kp_rotation = 1.5
        self.Ki_rotation = 0.1
        self.Kd_rotation = 0.3
        self.rotation_integral = 0.0
        self.last_rotation_error = 0.0
        
    def compute_motor_speeds(self, linear_speed, angular_speed):
        """Convertit les vitesses linéaire et angulaire en vitesses des moteurs"""
        # Calcul des vitesses des roues (gauche et droite)
        v_right = (linear_speed + angular_speed * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        v_left = (linear_speed - angular_speed * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        
        # Normalisation des vitesses
        max_speed = max(abs(v_right), abs(v_left))
        if max_speed > self.MAX_SPEED:
            v_right = v_right * self.MAX_SPEED / max_speed
            v_left = v_left * self.MAX_SPEED / max_speed
            
        return v_left, v_right

    def angle_difference(self, target_angle, current_angle):
        """Calcule la différence d'angle la plus courte"""
        diff = (target_angle - current_angle) % (2 * pi)
        if diff > pi:
            diff -= 2 * pi
        return diff

    def compute_pid_angle(self, target_angle, target_rotation, dt):
        """Calcule la correction PID pour l'angle et la rotation avec transition adaptative"""
        # PID pour l'angle de déplacement
        angle_error = self.angle_difference(target_angle, self.current_orientation)
        self.angle_integral = np.clip(self.angle_integral + angle_error * dt, -2.0, 2.0)
        angle_derivative = (angle_error - self.last_angle_error) / dt
        self.last_angle_error = angle_error
        
        angle_correction = (self.Kp_angle * angle_error +
                          self.Ki_angle * self.angle_integral +
                          self.Kd_angle * angle_derivative)
        
        # PID pour la rotation Z avec anti-windup
        rotation_error = self.angle_difference(target_rotation, self.current_orientation)
        self.rotation_integral = np.clip(self.rotation_integral + rotation_error * dt, -2.0, 2.0)
        rotation_derivative = (rotation_error - self.last_rotation_error) / dt
        self.last_rotation_error = rotation_error
        
        rotation_correction = (self.Kp_rotation * rotation_error +
                             self.Ki_rotation * self.rotation_integral +
                             self.Kd_rotation * rotation_derivative)
        
        # Calcul des poids adaptatifs basés sur les erreurs
        total_error = abs(angle_error) + abs(rotation_error)
        if total_error > 0:
            angle_weight = abs(angle_error) / total_error
            rotation_weight = abs(rotation_error) / total_error
        else:
            angle_weight = rotation_weight = 0.5

        # Transition douce entre mouvement et rotation
        movement_phase = np.exp(-abs(angle_error) * 2)  # Facteur de phase du mouvement
        rotation_phase = np.exp(-abs(rotation_error) * 2)  # Facteur de phase de rotation
        
        # Combinaison adaptative des corrections
        total_correction = (angle_weight * angle_correction * movement_phase +
                          rotation_weight * rotation_correction * rotation_phase)
        
        return total_correction

    def navigate_to_target(self, move_vector, target_orientation, distance):
        """Version améliorée avec gestion de phase et contrôle adaptatif"""
        # Calcul de l'angle de déplacement
        movement_angle = atan2(move_vector[1], move_vector[0])
        
        # Détermination de la phase de mouvement
        angle_error = abs(self.angle_difference(movement_angle, self.current_orientation))
        rotation_error = abs(self.angle_difference(target_orientation, self.current_orientation))
        
        # Paramètres de contrôle adaptatifs
        ROTATION_THRESHOLD = pi/6  # 30 degrés
        PRECISION_THRESHOLD = 0.2  # mètres
        
        # Temps d'échantillonnage
        dt = 0.02

        # Décision de phase de mouvement
        if angle_error > ROTATION_THRESHOLD:
            # Phase d'alignement
            angular_correction = self.compute_pid_angle(movement_angle, movement_angle, dt)
            linear_speed = self.MIN_SPEED * (1 - angle_error/pi)
        elif distance < PRECISION_THRESHOLD:
            # Phase d'orientation finale
            angular_correction = self.compute_pid_angle(target_orientation, target_orientation, dt)
            linear_speed = self.MIN_SPEED * (distance/PRECISION_THRESHOLD)
        else:
            # Phase de déplacement normal
            angular_correction = self.compute_pid_angle(movement_angle, target_orientation, dt)
            linear_speed = self.calculate_adaptive_speed(distance, angle_error, rotation_error)
        
        # Calcul final des vitesses
        v_left, v_right = self.compute_motor_speeds(linear_speed, angular_correction)
        
        # Mise à jour de la position estimée
        self.update_position(v_left, v_right, dt)
        
        return v_left, v_right

    def calculate_adaptive_speed(self, distance, angle_error, rotation_error):
        """Calcule une vitesse adaptative basée sur la situation"""
        base_speed = min(self.MAX_SPEED, max(self.MIN_SPEED, distance))
        
        # Facteurs de réduction
        angle_factor = np.exp(-angle_error * 2)
        rotation_factor = np.exp(-rotation_error * 2)
        distance_factor = np.clip(distance / 2.0, 0.1, 1.0)
        
        return base_speed * min(angle_factor, rotation_factor, distance_factor)

    def update_position(self, v_left, v_right, dt):
        """Met à jour la position et l'orientation estimées du robot"""
        # Calcul du déplacement
        linear_velocity = (v_right + v_left) * self.WHEEL_RADIUS / 2
        angular_velocity = (v_right - v_left) * self.WHEEL_RADIUS / self.WHEEL_BASE
        
        # Mise à jour de l'orientation
        self.current_orientation += angular_velocity * dt
        self.current_orientation = self.current_orientation % (2 * pi)
        
        # Mise à jour de la position
        dx = linear_velocity * cos(self.current_orientation) * dt
        dy = linear_velocity * sin(self.current_orientation) * dt
        self.current_pos += np.array([dx, dy])

    def emergency_stop(self):
        """Arrêt d'urgence du robot"""
        return 0, 0  # Vitesse nulle pour les deux moteurs

# Exemple d'utilisation:
def control_raspblock(strategic_move, timeout=30.0, precision=0.01):
    """
    Fonction principale de contrôle du Raspblock avec feedback et sécurité
    
    Args:
        strategic_move: tuple(np.array([dx, dy]), orientation, distance, analysis)
        timeout: float - Temps maximum d'exécution en secondes
        precision: float - Seuil de précision pour la position cible
    
    Returns:
        dict: État final du mouvement avec succès/échec et métriques
    """
    navigator = RaspblockNavigator()
    move_vector, orientation, distance, _ = strategic_move
    
    # État du mouvement
    state = {
        'start_time': time.time(),
        'last_progress': distance,
        'stuck_counter': 0,
        'adjustments': 0,
        'success': False
    }
    
    try:
        while distance > precision:
            # Vérification du timeout
            if time.time() - state['start_time'] > timeout:
                raise TimeoutError("Temps maximum d'exécution dépassé")
            
            # Obtenir les vitesses des moteurs
            v_left, v_right = navigator.navigate_to_target(move_vector, orientation, distance)
            
            # Commande aux moteurs avec vérification des limites
            v_left = np.clip(v_left, -navigator.MAX_SPEED, navigator.MAX_SPEED)
            v_right = np.clip(v_right, -navigator.MAX_SPEED, navigator.MAX_SPEED)
            
            try:
                # Envoi des commandes aux moteurs
                # raspblock.set_motor_speeds(v_left, v_right)
                print('raspblock.set_moto_speed(',v_left,v_right,')')
                pass  # À remplacer par l'appel réel aux moteurs
            except Exception as motor_error:
                print(f"Erreur moteurs: {motor_error}")
                navigator.emergency_stop()
                raise
            
            # Délai de contrôle adaptatif
            if distance < precision * 2:
                time.sleep(0.05)  # Plus lent près de la cible
            else:
                time.sleep(0.02)  # Normal
            
            # Mise à jour de la distance et vérification du progrès
            new_distance = np.linalg.norm(move_vector - navigator.current_pos)
            if abs(new_distance - state['last_progress']) < precision * 0.1:
                state['stuck_counter'] += 1
                if state['stuck_counter'] > 50:  # Bloqué pendant trop longtemps
                    # Tentative d'ajustement
                    state['adjustments'] += 1
                    if state['adjustments'] > 3:
                        raise RuntimeError("Robot bloqué - trop d'ajustements nécessaires")
                    # Petit mouvement aléatoire pour débloquer
                    angle_adjustment = np.random.uniform(-pi/4, pi/4)
                    v_left, v_right = navigator.compute_motor_speeds(0.1, angle_adjustment)
                    # raspblock.set_motor_speeds(v_left, v_right)
                    time.sleep(0.5)
                    state['stuck_counter'] = 0
            else:
                state['stuck_counter'] = 0
            
            state['last_progress'] = new_distance
            distance = new_distance
        
        # Mouvement terminé avec succès
        navigator.emergency_stop()
        state['success'] = True
        
    except Exception as e:
        print(f"Erreur de navigation: {e}")
        navigator.emergency_stop()
        state['error'] = str(e)
    finally:
        # Retour d'état complet
        state['final_distance'] = distance
        state['execution_time'] = time.time() - state['start_time']
        state['final_position'] = navigator.current_pos.copy()
        state['final_orientation'] = navigator.current_orientation
        
        return state

control_raspblock([[0,0],0,1,0], timeout=30.0, precision=0.01)