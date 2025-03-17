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
    # magnus = 0.5 * rho * np.linalg.norm(v) * np.cross(omega, v) * Cm * A / m  
    # return g + drag + magnus
    return g + drag

# Функция для симуляции полета мяча
def simulate(pos, v, r, m):
    global positions, velocities
    t = 0
    while pos[1] >= 0 and t < t_max:
        with data_lock:  # Защищаем доступ к данным
            positions.append(pos.copy())
            velocities.append(v.copy())

        v, pos = runge_kutta_4(v, pos, r, m, dt)
        t += dt
        
# Метод Рунге-Кутты 4-го порядка для рассчета скорости и позиции
def runge_kutta_4(v, pos, r, m, dt):
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

# Обновление графиков в реальном времени
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

# Построение графиков в реальном времени
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
    # run_opengl()

    # Построение графиков в реальном времени
    plot_graphs_real_time()

if __name__ == "__main__":
    main()
