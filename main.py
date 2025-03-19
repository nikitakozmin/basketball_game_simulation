import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


RHO = 1.225  # Плотность воздуха (кг/м³)
CD = 0.47  # Коэффициент сопротивления воздуха (для сферы)
CM = 0.2  # Коэффициент Магнуса
GRAVITY = np.array([0.0, 0.0, -9.81])  # Гравитация
BALL_RADIUS = 0.2
BALL_MASS = 0.6
FIELD_WIDTH = 10
FIELD_LENGTH = 5


# Функция для вычисления ускорения
def acceleration(v, r, m):
    A = np.pi * r**2  # Площадь поперечного сечения
    drag = -0.5 * RHO * np.linalg.norm(v) * v * CD * A / m
    # magnus = 0.5 * rho * np.linalg.norm(v) * np.cross(omega, v) * Cm * A / m  
    # return g + drag + magnus
    return GRAVITY + drag


# Явный метод Эйлера
def euler_method(ball_velocity, ball_position, dt):
    a = acceleration(ball_velocity, BALL_RADIUS, BALL_MASS)
    ball_velocity += a * dt
    ball_position += ball_velocity * dt
    return ball_velocity, ball_position


# Неявный метод Эйлера
def implicit_euler_method(dt, ball_velocity, ball_position):
    ball_velocity_next = ball_velocity.copy()
    for _ in range(10):  # Ограничиваем количество итераций
        # Вычисление ускорения на следующем шаге
        a_next = acceleration(ball_velocity_next, BALL_RADIUS, BALL_MASS)
        # Обновление скорости и позиции на следующем шаге
        ball_velocity_next_new = ball_velocity + a_next * dt
        ball_position_next = ball_position + ball_velocity_next_new * dt

        if np.linalg.norm(ball_velocity_next_new - ball_velocity_next) < 1e-6:
            break
        
        ball_velocity_next = ball_velocity_next_new

    return ball_velocity_next, ball_position_next
    

# Метод Рунге-Кутты 4-го порядка для рассчета скорости и позиции
def runge_kutta_4(ball_velocity, ball_position, ball_mass, dt):

    k1v = acceleration(ball_velocity, BALL_RADIUS, ball_mass) * dt
    k1x = ball_velocity * dt

    k2v = acceleration(ball_velocity + 0.5 * k1v, BALL_RADIUS, ball_mass) * dt
    k2x = (ball_velocity + 0.5 * k1v) * dt

    k3v = acceleration(ball_velocity + 0.5 * k2v, BALL_RADIUS, ball_mass) * dt
    k3x = (ball_velocity + 0.5 * k2v) * dt

    k4v = acceleration(ball_velocity + k3v, BALL_RADIUS, ball_mass) * dt
    k4x = (ball_velocity + k3v) * dt

    ball_velocity_new = ball_velocity + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    ball_position_new = ball_position + (k1x + 2 * k2x + 2 * k3x + k4x) / 6

    return ball_velocity_new, ball_position_new


# Адаптивный метод
def adaptive_step_method(ball_velocity, ball_position, dt):
    k1v = acceleration(ball_velocity, BALL_RADIUS, BALL_MASS) * dt
    k1x = ball_velocity * dt
    k2v = acceleration(ball_velocity + 0.5 * k1v, BALL_RADIUS, BALL_MASS) * dt
    k2x = (ball_velocity + 0.5 * k1v) * dt
    k3v = acceleration(ball_velocity + 0.5 * k2v, BALL_RADIUS, BALL_MASS) * dt
    k3x = (ball_velocity + 0.5 * k2v) * dt
    k4v = acceleration(ball_velocity + k3v, BALL_RADIUS, BALL_MASS) * dt
    k4x = (ball_velocity + k3v) * dt

    ball_velocity_new = ball_velocity + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    ball_position_new = ball_position + (k1x + 2 * k2x + 2 * k3x + k4x) / 6

    error = np.linalg.norm(ball_position_new - ball_position)  # Оценка ошибки
    if error < 1e-3:  # Если ошибка достаточно мала
        dt *= 1.5  # Увеличиваем шаг
    else:
        dt *= 0.5  # Уменьшаем шаг
    return ball_velocity_new, ball_position_new


# Предиктор-корректор
def predictor_corrector_method(ball_velocity, ball_position, dt):
    a = acceleration(ball_velocity, BALL_RADIUS, BALL_MASS)
    ball_velocity_pred = ball_velocity + a * dt
    # ball_position_pred = ball_position + ball_velocity * dt

    # Коррекция с использованием Рунге-Кутты 2-го порядка (для примера)
    k1v = acceleration(ball_velocity_pred, BALL_RADIUS, BALL_MASS) * dt
    k1x = ball_velocity_pred * dt
    k2v = acceleration(ball_velocity_pred + 0.5 * k1v, BALL_RADIUS, BALL_MASS) * dt
    k2x = (ball_velocity_pred + 0.5 * k1v) * dt

    ball_velocity_new = ball_velocity + (k1v + k2v) / 2
    ball_position_new = ball_position + (k1x + k2x) / 2

    return ball_velocity_new, ball_position_new


def choose_method():
    print("Выберите метод расчета:")
    print("1. Явный метод Эйлера")
    print("2. Неявный метод Эйлера")
    print("3. Метод Рунге-Кутты 4-го порядка")
    print("4. Метод предиктор-корректор")
    print("5. Адаптивный метод")
    choice = int(input("Введите номер метода: "))
    return choice


# Функция для рисования сферы (мячика)
def draw_sphere(pos, radius, slices=16, stacks=16):
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, slices, stacks)
    gluDeleteQuadric(quadric)
    glPopMatrix()


def init_visual_OpenGL():
    # Инициализация Pygame и OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Настройка камеры и перспективы
    gluPerspective(45, (display[0] / display[1]), 0.1, 200.0)
    glTranslatef(0.0, 0.0, -20)  # Отодвинули камеру на 20 единиц

    # Включение теста глубины
    glEnable(GL_DEPTH_TEST)


def update_visual_OpenGL(camera_rotation):
    # Очистка экрана и буфера глубины
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Применение вращения камеры
    glPushMatrix()
    glRotatef(camera_rotation[0], 1, 0, 0)  # Вращение по X
    glRotatef(camera_rotation[1], 0, 1, 0)  # Вращение по Y

    # Рисование земли (плоскости)
    glBegin(GL_QUADS)
    glColor3f(0.5, 0.5, 0.5)  # Серый цвет
    glVertex3f(-FIELD_WIDTH/2, -FIELD_LENGTH/2, 0)
    glVertex3f(FIELD_WIDTH-FIELD_WIDTH/2, -FIELD_LENGTH/2, 0)
    glVertex3f(FIELD_WIDTH-FIELD_WIDTH/2, FIELD_LENGTH-FIELD_LENGTH/2, 0)
    glVertex3f(-FIELD_WIDTH/2, FIELD_LENGTH-FIELD_LENGTH/2, 0)
    glEnd()


def init_visual_graphs():
    plt.ion()  # Включение интерактивного режима
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.5)
    ax1.set_title('Траектория по X')
    ax2.set_title('Траектория по Y')
    ax3.set_title('Траектория по Z')
    ax3.set_xlabel('Время')
    line_x, = ax1.plot([], [], 'r-')
    line_y, = ax2.plot([], [], 'g-')
    line_z, = ax3.plot([], [], 'b-')
    fig.canvas.draw()
    fig.canvas.flush_events()
    return ax1, ax2, ax3, line_x, line_y, line_z, fig


def update_plot(time_steps, positions, line_x, line_y, line_z, ax1, ax2, ax3, fig):
    line_x.set_data(time_steps, [p[0] for p in positions])
    line_y.set_data(time_steps, [p[1] for p in positions])
    line_z.set_data(time_steps, [p[2] for p in positions])
    
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()

    fig.canvas.draw()  # Быстрое обновление графиков
    fig.canvas.flush_events()


def handle_events(running, mouse_dragging, last_mouse_pos, camera_rotation):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Управление камерой с помощью мыши
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Левая кнопка мыши
                mouse_dragging = True
                last_mouse_pos = [event.pos[0], event.pos[1]]
            if event.button == 4:  # Колёсико вверх
                glTranslatef(0.0, 0.0, 1.0)  # Приблизить камеру
            if event.button == 5:  # Колёсико вниз
                glTranslatef(0.0, 0.0, -1.0)  # Отдалить камеру

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Левая кнопка мыши
                mouse_dragging = False

        if event.type == pygame.MOUSEMOTION:
            if mouse_dragging:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                camera_rotation[0] += dy * 0.1  # Вращение по X
                camera_rotation[1] += dx * 0.1  # Вращение по Y
                last_mouse_pos = [event.pos[0], event.pos[1]]
    return running, mouse_dragging, last_mouse_pos, camera_rotation


def main():
    method_choice = choose_method()

    init_visual_OpenGL()

    # Управление камерой
    mouse_dragging = False
    last_mouse_pos = [0, 0]
    camera_rotation = [0, 0]  # Вращение камеры (X, Y)
    ball_pos = np.array([0, 0, BALL_RADIUS])  # Начальная позиция
    
    update_visual_OpenGL(camera_rotation)
    glColor3f(1.0, 0.0, 0.0)  # Красный цвет
    draw_sphere(ball_pos, BALL_RADIUS)
    
    glPopMatrix()  # Возвращаем камеру в исходное состояние
    
    pygame.display.flip()
    
    ball_velocity = np.array([float(input("Введите скорость по ширине (по x): ")), float(input("Введите скорость по ширине (по y): ")), float(input("Введите скорость по ширине (по z): "))])  # Начальная скорость     
    omega = np.array([0, 0, 0])  # Угловая скорость (рад/с) — убираем вращение
    
    dt = 0.01  # Шаг по времени
    
    # Данные для графиков
    positions = []  # Список для хранения позиций мячика
    time_steps = []  # Список для хранения времени
    current_time = 0  # Текущее время
    ax1, ax2, ax3, line_x, line_y, line_z, fig = init_visual_graphs()

    # Основной цикл
    clock = pygame.time.Clock()
    running = True
    frames_since_last_plot_update = 0
    plot_update_interval = 10

    while running:
        running, mouse_dragging, last_mouse_pos, camera_rotation = handle_events(running, mouse_dragging, last_mouse_pos, camera_rotation)

        update_visual_OpenGL(camera_rotation)
        
        if method_choice == 1:
            # Явный метод Эйлера
            ball_velocity, ball_pos = euler_method(ball_velocity, ball_pos, dt)
        elif method_choice == 2:
            # Неявный метод Эйлера
            ball_velocity, ball_pos = implicit_euler_method(dt, ball_velocity, ball_pos)
        elif method_choice == 3:
            # Метод Рунге-Кутты 4 порядка
            ball_velocity, ball_pos = runge_kutta_4(ball_velocity, ball_pos, BALL_MASS, dt)
        elif method_choice == 4:
            # Метод предиктор-корректор
            ball_velocity, ball_pos = predictor_corrector_method(ball_velocity, ball_pos, dt)
        elif method_choice == 5:
            # Адаптивный метод
            ball_velocity, ball_pos = adaptive_step_method(ball_velocity, ball_pos, dt)

        # Отскок от земли (если мячик падает ниже z = 0)
        if ball_pos[2] - BALL_RADIUS < 0:
            ball_pos[2] = BALL_RADIUS  # Поднимаем мячик над землёй
            ball_velocity[2] = -ball_velocity[2] * 0.8  # Потеря энергии при отскоке

        # Отскок от стен (по X и Y)
        if ball_pos[0] - BALL_RADIUS < -FIELD_WIDTH/2 or ball_pos[0] + BALL_RADIUS > FIELD_WIDTH - FIELD_WIDTH/2:
            ball_velocity[0] = -ball_velocity[0]  # Отскок по X
        if ball_pos[1] - BALL_RADIUS < -FIELD_LENGTH/2 or ball_pos[1] + BALL_RADIUS > FIELD_LENGTH - FIELD_LENGTH/2:
            ball_velocity[1] = -ball_velocity[1]  # Отскок по Y

        # Рисование мячика
        glColor3f(1.0, 0.0, 0.0)  # Красный цвет
        draw_sphere(ball_pos, BALL_RADIUS)
        
        glPopMatrix()  # Возвращаем камеру в исходное состояние

        # Сохранение данных для графиков
        positions.append(ball_pos.copy())
        time_steps.append(current_time)
        current_time += 1

        # Обновление графиков через каждые N кадров
        frames_since_last_plot_update += 1
        if frames_since_last_plot_update >= plot_update_interval:
            update_plot(time_steps, positions, line_x, line_y, line_z, ax1, ax2, ax3, fig)
            frames_since_last_plot_update = 0

        # Обновление экрана
        pygame.display.flip()
        clock.tick(60) 

    pygame.quit()


if __name__ == "__main__":
    main()
