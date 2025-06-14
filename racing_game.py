import pygame
import pymunk
import math
import sys
import torch
from rl_agent import RLAgent
import numpy as np
import os
import glob

# --- RL Constants ---
NUM_LIDAR_RAYS = 8
STATE_DIM = NUM_LIDAR_RAYS + 4  # Lidar distances + velocity, ang_vel, dist_to_next, angle_to_next
ACTION_DIM = 9  # 0:No-op, 1:Up, 2:Down, 3:Left, 4:Right, 5:UpL, 6:UpR, 7:DownL, 8:DownR

# New screen size
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
space = pymunk.Space()
space.gravity = (0, 0)
font = pygame.font.SysFont(None, 48)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (60, 60, 60)
DARK_GRAY = (30, 30, 30)
GREEN = (34, 139, 34)
YELLOW = (255, 255, 0)
SHADOW = (0, 0, 0, 80)
TIRE_MARK = (40, 40, 40)
START_LINE_COLOR = (0, 200, 255)
CHECK_LINE_COLOR = (255, 200, 0)

# More complex track polygons (fit to new screen size)
OUTER_TRACK = [
    (200, 150), (1200, 150), (1300, 300), (1200, 750), (200, 750), (100, 600), (100, 300)
]
INNER_TRACK = [
    (400, 350), (1100, 350), (1150, 450), (1100, 650), (400, 650), (250, 600), (250, 400)
]

# Tire marks
tire_marks = []

# Start/finish line and checkpoints (as line segments: ((x1, y1), (x2, y2)))
START_LINE = ((200, 150), (400, 350))

# Generate 8 evenly spaced checkpoints between OUTER_TRACK and INNER_TRACK
NUM_CHECKPOINTS = 8

def interpolate_polygon(poly, num_points):
    # Returns a list of num_points evenly spaced along the polygon
    points = []
    n = len(poly)
    # Calculate total perimeter
    dists = []
    for i in range(n):
        a, b = poly[i], poly[(i+1)%n]
        dists.append(math.hypot(b[0]-a[0], b[1]-a[1]))
    perimeter = sum(dists)
    step = perimeter / num_points
    curr_dist = 0
    seg_idx = 0
    seg_pos = 0
    for i in range(num_points):
        while seg_pos + dists[seg_idx] < curr_dist:
            seg_pos += dists[seg_idx]
            seg_idx = (seg_idx + 1) % n
        a = poly[seg_idx]
        b = poly[(seg_idx+1)%n]
        seg_len = dists[seg_idx]
        t = (curr_dist - seg_pos) / seg_len if seg_len > 0 else 0
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        points.append((x, y))
        curr_dist += step
    return points

OUTER_POINTS = interpolate_polygon(OUTER_TRACK, NUM_CHECKPOINTS)
INNER_POINTS = interpolate_polygon(INNER_TRACK, NUM_CHECKPOINTS)
CHECK_LINES = list(zip(OUTER_POINTS, INNER_POINTS))

score = 0
checkpoints_crossed = set()

def point_in_polygon(point, polygon):
    # Ray casting algorithm for point in polygon
    x, y = point
    inside = False
    n = len(polygon)
    px1, py1 = polygon[0]
    for i in range(n+1):
        px2, py2 = polygon[i % n]
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1 + 1e-10) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2
    return inside

class Car:
    def __init__(self, x, y):
        self.width = 40
        self.height = 20
        self.body = pymunk.Body(1, 100)
        self.body.position = x, y
        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        self.shape.friction = 0.7
        self.shape.elasticity = 0.5
        space.add(self.body, self.shape)
        self.velocity = pygame.Vector2(0, 0)
        self.angle = -90  # Facing up
        self.acceleration = 0
        self.max_speed = 10
        self.steering = 0
        self.steering_speed = 3.5  # degrees per frame
        self.acceleration_speed = 0.25
        self.brake_speed = 0.4
        self.friction = 0.04
        self.drift_factor = 0.92  # Lower = more drift
        self.last_pos = pygame.Vector2(x, y)
        self.last_cross_dir = None  # Track last crossing direction
        self.collided = False

    def update(self):
        # Handle steering
        if self.steering != 0 and self.velocity.length() > 0.5:
            steer_amount = self.steering * self.steering_speed * (self.velocity.length() / self.max_speed)
            self.angle += steer_amount
        # Handle acceleration/braking
        direction = pygame.Vector2(math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle)))
        if self.acceleration > 0:
            self.velocity += direction * self.acceleration_speed
        elif self.acceleration < 0:
            self.velocity -= direction * self.brake_speed
        # Friction
        self.velocity *= (1 - self.friction)
        # Limit speed
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        # Drifting: blend velocity toward car's facing direction
        if self.velocity.length() > 0.5:
            forward = direction * self.velocity.dot(direction)
            lateral = self.velocity - forward
            self.velocity = forward + lateral * self.drift_factor
        # Move car
        prev_pos = pygame.Vector2(self.body.position.x, self.body.position.y)
        self.body.position += pymunk.Vec2d(self.velocity.x, self.velocity.y)
        # Collision with track walls (stay between outer and inner track)
        car_center = (self.body.position.x, self.body.position.y)
        inside_outer = point_in_polygon(car_center, OUTER_TRACK)
        inside_inner = point_in_polygon(car_center, INNER_TRACK)
        if not inside_outer or inside_inner:
            self.body.position = pymunk.Vec2d(prev_pos.x, prev_pos.y)
            self.velocity *= -0.3
            self.collided = True
        # Check for start/finish and checkpoints
        self.handle_lines(prev_pos, self.body.position)
        # Update body angle for drawing
        self.body.angle = math.radians(self.angle)
        # Tire marks if drifting
        if self.velocity.length() > 4 and abs(self.steering) > 0 and lateral.length() > 1.5:
            tire_marks.append((self.body.position.x, self.body.position.y, self.body.angle))
        self.last_pos = pygame.Vector2(self.body.position.x, self.body.position.y)

    def handle_lines(self, prev_pos, curr_pos):
        global score, checkpoints_crossed
        # Only count start/finish crossing if moving in the correct direction
        # Calculate direction vector of the start line
        sx1, sy1 = START_LINE[0]
        sx2, sy2 = START_LINE[1]
        start_vec = pygame.Vector2(sx2 - sx1, sy2 - sy1).normalize()
        # Car movement vector
        move_vec = pygame.Vector2(curr_pos.x - prev_pos.x, curr_pos.y - prev_pos.y)
        if move_vec.length() > 0:
            move_dir = move_vec.normalize()
        else:
            move_dir = pygame.Vector2(0, 0)
        # Perpendicular to start line (which side is "forward")
        perp = pygame.Vector2(-start_vec.y, start_vec.x)
        # Loosen the direction check (adjust threshold or flip sign if needed)
        correct_direction = move_dir.dot(perp) > 0.0  # Try 0.0 or -0.1 if needed
        # Start/finish line
        if lines_crossed((prev_pos.x, prev_pos.y), (curr_pos.x, curr_pos.y), *START_LINE):
            if correct_direction and len(checkpoints_crossed) == len(CHECK_LINES):
                score += 10  # Bonus for full lap
                checkpoints_crossed = set()
        # Checkpoints
        for i, (a, b) in enumerate(CHECK_LINES):
            if i not in checkpoints_crossed and lines_crossed((prev_pos.x, prev_pos.y), (curr_pos.x, curr_pos.y), a, b):
                score += 1
                checkpoints_crossed.add(i)

    def draw(self, screen):
        pos = self.body.position
        angle = self.body.angle
        shadow_surface = pygame.Surface((self.width+10, self.height+10), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, SHADOW, (0, 0, self.width+10, self.height+10))
        rotated_shadow = pygame.transform.rotate(shadow_surface, -math.degrees(angle))
        screen.blit(rotated_shadow, (pos.x - rotated_shadow.get_width()/2 + 4, pos.y - rotated_shadow.get_height()/2 + 8))
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, RED, (0, 0, self.width, self.height), border_radius=8)
        pygame.draw.rect(car_surface, YELLOW, (self.width-10, 5, 8, 10), border_radius=3)
        pygame.draw.rect(car_surface, DARK_GRAY, (2, 2, self.width-4, self.height-4), 2)
        pygame.draw.rect(car_surface, WHITE, (10, 4, 18, 12), border_radius=4)
        rotated_car = pygame.transform.rotate(car_surface, -math.degrees(angle))
        screen.blit(rotated_car, (pos.x - rotated_car.get_width()/2, pos.y - rotated_car.get_height()/2))

def draw_background():
    screen.fill(GREEN)
    pygame.draw.polygon(screen, GRAY, OUTER_TRACK)
    pygame.draw.polygon(screen, GREEN, INNER_TRACK)
    # Draw start/finish line
    pygame.draw.line(screen, START_LINE_COLOR, *START_LINE, 8)
    # Draw checkpoints
    for a, b in CHECK_LINES:
        pygame.draw.line(screen, CHECK_LINE_COLOR, a, b, 5)

def draw_track_walls():
    pygame.draw.lines(screen, BLACK, True, OUTER_TRACK, 4)
    pygame.draw.lines(screen, BLACK, True, INNER_TRACK, 4)

def draw_tire_marks():
    for mark in tire_marks[-800:]:
        x, y, angle = mark
        end_x = x - 8 * math.cos(angle + 0.2)
        end_y = y - 8 * math.sin(angle + 0.2)
        pygame.draw.line(screen, TIRE_MARK, (x, y), (end_x, end_y), 2)
        end_x2 = x - 8 * math.cos(angle - 0.2)
        end_y2 = y - 8 * math.sin(angle - 0.2)
        pygame.draw.line(screen, TIRE_MARK, (x, y), (end_x2, end_y2), 2)

def create_track():
    for i in range(len(OUTER_TRACK)):
        a = OUTER_TRACK[i]
        b = OUTER_TRACK[(i+1)%len(OUTER_TRACK)]
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, a, b, 2)
        shape.friction = 0.7
        shape.elasticity = 0.5
        space.add(body, shape)
    for i in range(len(INNER_TRACK)):
        a = INNER_TRACK[i]
        b = INNER_TRACK[(i+1)%len(INNER_TRACK)]
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, a, b, 2)
        shape.friction = 0.7
        shape.elasticity = 0.5
        space.add(body, shape)

def draw_score():
    score_surf = font.render(f"Score: {score}", True, (0,0,0))
    screen.blit(score_surf, (30, 30))
    mx, my = pygame.mouse.get_pos()
    mouse_surf = font.render(f"Mouse: ({mx}, {my})", True, (0,0,0))
    screen.blit(mouse_surf, (30, 80))

def lines_crossed(p1, p2, l1, l2):
    # Returns True if line (p1,p2) crosses (l1,l2)
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return (ccw(p1, l1, l2) != ccw(p2, l1, l2)) and (ccw(p1, p2, l1) != ccw(p1, p2, l2))

def main():
    global score, checkpoints_crossed
    car = Car(200, 250)
    create_track()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        car.acceleration = 0
        car.steering = 0
        if keys[pygame.K_UP]:
            car.acceleration = 1
        if keys[pygame.K_DOWN]:
            car.acceleration = -1
        if keys[pygame.K_LEFT]:
            car.steering = 1
        if keys[pygame.K_RIGHT]:
            car.steering = -1
        car.update()
        space.step(1/60)
        draw_background()
        draw_tire_marks()
        draw_track_walls()
        car.draw(screen)
        draw_score()
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    sys.exit()

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.init_game_objects()

    def init_game_objects(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.car = Car(700, 250)
        self.create_track()
        self.score = 0
        self.checkpoints_crossed = set()
        self.next_checkpoint_idx = 0
        # Orient car towards the center of the first checkpoint
        cp = CHECK_LINES[0]
        cp_center = ((cp[0][0] + cp[1][0]) / 2, (cp[0][1] + cp[1][1]) / 2)
        car_pos = self.car.body.position
        vec_to_cp = pymunk.Vec2d(cp_center[0] - car_pos.x, cp_center[1] - car_pos.y)
        angle = math.atan2(vec_to_cp.y, vec_to_cp.x)
        self.car.angle = math.degrees(angle)
        self.car.body.angle = angle

    def create_track(self):
        # ... (keep existing track creation logic) ...
        self.track_walls = []
        for i in range(len(OUTER_TRACK)):
            a, b = OUTER_TRACK[i], OUTER_TRACK[(i+1)%len(OUTER_TRACK)]
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, a, b, 2)
            self.space.add(body, shape)
            self.track_walls.append(shape)
        for i in range(len(INNER_TRACK)):
            a, b = INNER_TRACK[i], INNER_TRACK[(i+1)%len(INNER_TRACK)]
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, a, b, 2)
            self.space.add(body, shape)
            self.track_walls.append(shape)

    def get_lidar_readings(self):
        readings = []
        for i in range(NUM_LIDAR_RAYS):
            angle = self.car.body.angle + (i * 2 * math.pi / NUM_LIDAR_RAYS)
            end_point = self.car.body.position + pymunk.Vec2d(math.cos(angle), math.sin(angle)) * 1000
            segment_query = self.space.segment_query_first(self.car.body.position, end_point, 1, pymunk.ShapeFilter())
            if segment_query:
                distance = (segment_query.point - self.car.body.position).length
                readings.append(distance / 1000.0) # Normalize
            else:
                readings.append(1.0)
        return readings

    def get_state(self):
        lidar = self.get_lidar_readings()
        vel = self.car.velocity.length() / self.car.max_speed # Normalize
        ang_vel = self.car.body.angular_velocity / 10.0 # Normalize

        next_cp = CHECK_LINES[self.next_checkpoint_idx]
        cp_center = ((next_cp[0][0] + next_cp[1][0])/2, (next_cp[0][1] + next_cp[1][1])/2)
        car_pos = self.car.body.position
        
        vec_to_cp = pymunk.Vec2d(cp_center[0] - car_pos.x, cp_center[1] - car_pos.y)
        dist_to_cp = vec_to_cp.length / 1500.0 # Normalize

        car_dir = pymunk.Vec2d(math.cos(self.car.body.angle), math.sin(self.car.body.angle))
        angle_to_cp = car_dir.get_angle_between(vec_to_cp) / math.pi # Normalize

        return torch.FloatTensor(lidar + [vel, ang_vel, dist_to_cp, angle_to_cp]).unsqueeze(0)

    def _distance_to_next_checkpoint(self, pos=None):
        if pos is None:
            pos = self.car.body.position
        next_cp = CHECK_LINES[self.next_checkpoint_idx]
        cp_center = ((next_cp[0][0] + next_cp[1][0])/2, (next_cp[0][1] + next_cp[1][1])/2)
        return math.hypot(cp_center[0] - pos.x, cp_center[1] - pos.y)

    def step(self, action):
        # Map action index to car controls
        # ... (logic for actions 0-8) ...
        if action == 1: self.car.acceleration = 1
        elif action == 2: self.car.acceleration = -1
        if action == 3: self.car.steering = 1
        elif action == 4: self.car.steering = -1
        if action == 5: self.car.acceleration, self.car.steering = 1, 1
        elif action == 6: self.car.acceleration, self.car.steering = 1, -1
        if action == 7: self.car.acceleration, self.car.steering = -1, 1
        elif action == 8: self.car.acceleration, self.car.steering = -1, -1

        prev_pos = pygame.Vector2(self.car.body.position.x, self.car.body.position.y)
        prev_dist = self._distance_to_next_checkpoint(prev_pos)
        self.car.update()
        self.space.step(1/60)
        new_pos = pygame.Vector2(self.car.body.position.x, self.car.body.position.y)
        new_dist = self._distance_to_next_checkpoint(new_pos)
        progress_reward = (prev_dist - new_dist) * 20  # much more valuable

        # Calculate reward
        reward = -0.1 + progress_reward # Time penalty + progress
        done = False

        # Collision check
        if self.car.collided:
            reward -= 20
            done = True # End episode on crash
            self.car.collided = False
        
        # Checkpoint reward
        if lines_crossed((prev_pos.x, prev_pos.y), (self.car.body.position.x, self.car.body.position.y), *CHECK_LINES[self.next_checkpoint_idx]):
            reward += 100  # much more valuable
            self.checkpoints_crossed.add(self.next_checkpoint_idx)
            self.next_checkpoint_idx = (self.next_checkpoint_idx + 1) % len(CHECK_LINES)
        
        # Lap reward
        if lines_crossed((prev_pos.x, prev_pos.y), (self.car.body.position.x, self.car.body.position.y), *START_LINE):
            if len(self.checkpoints_crossed) >= len(CHECK_LINES) / 2: # Crossed enough checkpoints
                reward += 100
                self.checkpoints_crossed = set()
        
        self.car.acceleration = 0
        self.car.steering = 0

        next_state = self.get_state()
        return torch.tensor([reward], dtype=torch.float32), next_state, torch.tensor([done])

    def reset(self):
        self.init_game_objects()
        return self.get_state()

    def render(self, episode, total_reward):
        # Process pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        draw_background()
        draw_tire_marks()
        draw_track_walls()
        self.car.draw(self.screen)
        draw_score()
        info = self.font.render(f"Ep: {episode}  Reward: {total_reward:.1f}", True, (0,0,0))
        self.screen.blit(info, (30, 130))
        pygame.display.flip()
        self.clock.tick(60)

# --- Main RL Training Loop ---
def train_rl_agent():
    game = Game()
    agent = RLAgent(STATE_DIM, ACTION_DIM)

    # Load the latest saved model if available
    model_files = sorted(glob.glob("dqn_model_ep*.pth"), key=lambda x: int(x.split("_ep")[-1].split(".pth")[0]))
    if model_files:
        print(f"Loading model from {model_files[-1]}")
        checkpoint = torch.load(model_files[-1])
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if 'steps_done' in checkpoint:
            agent.steps_done = checkpoint['steps_done']

    num_episodes = 1000
    for i_episode in range(num_episodes):
        state = game.reset()
        total_reward = 0
        for t in range(2000): # Max steps per episode
            action = agent.select_action(state)
            reward, next_state, done = game.step(action.item())
            total_reward += reward.item()

            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.optimize_model()
            
            if done:
                break
        
        if i_episode % agent.target_update == 0:
            agent.update_target_net()

        print(f"Episode {i_episode}, Total Reward: {total_reward:.2f}")

        # Save the model every 100th episode
        if i_episode % 100 == 0:
            torch.save({
                'model_state_dict': agent.policy_net.state_dict(),
                'steps_done': agent.steps_done
            }, f"dqn_model_ep{i_episode}.pth")

        # Render every 20th episode, and make rendered episodes last up to 2200 steps (as user changed)
        if i_episode % 20 == 0:
            state = game.reset()
            for t in range(2200):
                action = agent.select_action(state)
                _, next_state, done = game.step(action.item())
                game.render(i_episode, total_reward)
                state = next_state
                if done:
                    break

if __name__ == "__main__":
    # To train the agent:
    train_rl_agent()
    
    # To play the game manually (you'll need to re-add the old main loop)
    # main() 