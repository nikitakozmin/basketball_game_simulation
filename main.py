import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

# --- Физические параметры ---
g = np.array([0, -9.81, 0])  # Гравитация
rho = 1.225  # Плотность воздуха (кг/м³)
Cd = 0.47  # Коэффициент сопротивления воздуха (для сферы)
Cm = 0.2  # Коэффициент Магнуса

# --- Начальные условия ---
dt = 0.01  # Шаг по времени
t_max = 3  # Время симуляции
omega = np.array([0, 0, 0])  # Угловая скорость (рад/с) — убираем вращение

# --- Хранение данных для графиков ---
positions = []
velocities = []

# Создаем блокировку для синхронизации доступа к данным
data_lock = threading.Lock()

# Функция для вычисления ускорения
def acceleration(v, r, m):
    A = np.pi * r**2  # Площадь поперечного сечения
    drag = -0.5 * rho * np.linalg.norm(v) * v * Cd * A / m
    magnus = 0.5 * rho * np.linalg.norm(v) * np.cross(omega, v) * Cm * A / m
    return g + drag + magnus

# --- Функция для симуляции полета мяча ---
def simulate(pos, v, r, m):
    global positions, velocities
    t = 0
    while pos[1] >= 0 and t < t_max:
        with data_lock:  # Защищаем доступ к данным
            positions.append(pos.copy())
            velocities.append(v.copy())

        v, pos = runge_kutta_4(v, pos, r, m, dt)
        t += dt

def runge_kutta_4(v, pos, r, m, dt):
    # Метод Рунге-Кутты 4-го порядка
    k1v = acceleration(v, r, m) * dt
    k1x = v * dt

    k2v = acceleration(v + 0.5 * k1v, r, m) * dt
    k2x = (v + 0.5 * k1v) * dt

    k3v = acceleration(v + 0.5 * k2v, r, m) * dt
    k3x = (v + 0.5 * k2v) * dt

    k4v = acceleration(v + k3v, r, m) * dt
    k4x = (v + k3v) * dt

    v_new = v + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    pos_new = pos + (k1x + 2 * k2x + 2 * k3x + k4x) / 6

    return v_new, pos_new

# --- OpenGL Визуализация ---
def init_opengl():
    glutInit()  # Инициализация GLUT
    pygame.init()
    width, height = 800, 600
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)  # Включаем тест глубины
    gluPerspective(45, width / height, 0.1, 50.0)  # Устанавливаем перспективу
    glTranslatef(0, 0, -10)  # Камера отодвигается на 10 единиц по оси Z

    # Освещение для лучшего отображения объектов
    glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
def draw_ball(x, y, z, r):
    glPushMatrix()
    glTranslatef(x, y, z)
    glColor3f(1, 0, 0)  # Красный цвет
    glutSolidSphere(r, 20, 20)
    glPopMatrix()

def draw_ground():
    glPushMatrix()
    glColor3f(0.5, 0.5, 0.5)  # Цвет земли
    glBegin(GL_QUADS)
    glVertex3f(-10, 0, -10)
    glVertex3f(10, 0, -10)
    glVertex3f(10, 0, 10)
    glVertex3f(-10, 0, 10)
    glEnd()
    glPopMatrix()

def draw_ring():
    glPushMatrix()
    glTranslatef(0, 3, 0)  # Позиция кольца
    glColor3f(0.5, 0.5, 0.5)  # Цвет кольца
    glutSolidTorus(0.05, 0.5, 50, 50)  # Нарисовать кольцо
    glPopMatrix()

# --- Графический интерфейс для ввода параметров ---
class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = (255, 255, 255)
        self.text = text
        self.font = pygame.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 2)
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))

def draw_text(screen, text, x, y, size=24, color=(255, 255, 255)):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def run_opengl():
    init_opengl()

    # Создаем текстовые поля для ввода параметров
    input_boxes = [
        InputBox(50, 50, 140, 32, "0"),  # Начальная координата X
        InputBox(50, 100, 140, 32, "1"),  # Начальная координата Y
        InputBox(50, 150, 140, 32, "5"),  # Начальная скорость по X
        InputBox(50, 200, 140, 32, "10"),  # Начальная скорость по Y
        InputBox(50, 250, 140, 32, "0.62"),  # Масса мяча
        InputBox(50, 300, 140, 32, "0.5"),  # Радиус мяча
    ]
    labels = [
        "Начальная координата X:",
        "Начальная координата Y:",
        "Начальная скорость по X:",
        "Начальная скорость по Y:",
        "Масса мяча (кг):",
        "Радиус мяча (м):",
    ]

    running = True
    simulation_started = False
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            for box in input_boxes:
                box.handle_event(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Запуск симуляции при нажатии Enter
                    pos = np.array([float(input_boxes[0].text), float(input_boxes[1].text), 0])
                    v = np.array([float(input_boxes[2].text), float(input_boxes[3].text), 0])
                    m = float(input_boxes[4].text)
                    r = float(input_boxes[5].text)
                    simulation_started = True
                    sim_thread = threading.Thread(target=simulate, args=(pos, v, r, m))
                    sim_thread.start()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if simulation_started:
            with data_lock:  # Защищаем доступ к данным
                if positions:
                    x, y, z = positions[-1]  # Берем последнюю позицию мяча
                    draw_ball(x, y, z, r)

        draw_ground()
        draw_ring()

        # Отрисовка текстовых полей и подписей
        screen = pygame.display.get_surface()
        for i, box in enumerate(input_boxes):
            draw_text(screen, labels[i], 50, box.rect.y - 25)
            box.draw(screen)

        pygame.display.flip()
        glFlush()

        clock.tick(60)  # Управление частотой кадров (60 FPS)

    pygame.quit()

# --- Построение графиков в реальном времени ---
def update_graphs(i, line_pos, line_vel, line_time, ax_pos, ax_vel, ax_time):
    with data_lock:  # Защищаем доступ к данным
        positions_np = np.array(positions)
        velocities_np = np.array(velocities)

    line_pos.set_data(positions_np[:i, 0], positions_np[:i, 1])
    line_vel.set_data(np.arange(i) * dt, velocities_np[:i, 1])
    line_time.set_data(np.arange(i) * dt, positions_np[:i, 1])

    ax_pos.set_xlim(np.min(positions_np[:, 0]) - 1, np.max(positions_np[:, 0]) + 1)
    ax_pos.set_ylim(np.min(positions_np[:, 1]) - 1, np.max(positions_np[:, 1]) + 1)
    ax_vel.set_xlim(0, np.max(i * dt) + 1)
    ax_vel.set_ylim(np.min(velocities_np[:, 1]) - 1, np.max(velocities_np[:, 1]) + 1)
    ax_time.set_xlim(0, np.max(i * dt) + 1)
    ax_time.set_ylim(np.min(positions_np[:, 1]) - 1, np.max(positions_np[:, 1]) + 1)

    return line_pos, line_vel, line_time

def plot_graphs_real_time():
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    ax_pos = axs[0]
    ax_pos.set_title('Траектория мяча')
    ax_pos.set_xlabel('X (м)')
    ax_pos.set_ylabel('Y (м)')

    ax_vel = axs[1]
    ax_vel.set_title('Скорость мяча по Y')
    ax_vel.set_xlabel('Время (с)')
    ax_vel.set_ylabel('Скорость (м/с)')

    ax_time = axs[2]
    ax_time.set_title('Позиция мяча по Y')
    ax_time.set_xlabel('Время (с)')
    ax_time.set_ylabel('Позиция (м)')

    line_pos, = ax_pos.plot([], [], label="Траектория (X-Y)")
    line_vel, = ax_vel.plot([], [], label="Скорость по Y")
    line_time, = ax_time.plot([], [], label="Позиция по Y")

    ani = FuncAnimation(fig, update_graphs, len(positions), fargs=(line_pos, line_vel, line_time, ax_pos, ax_vel, ax_time), interval=10)
    plt.tight_layout()
    plt.show()

def main():
    # Запуск OpenGL в основном потоке
    run_opengl()

    # Построение графиков в реальном времени
    plot_graphs_real_time()

if __name__ == "__main__":
    main()
