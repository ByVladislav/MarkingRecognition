import cv2
import matplotlib.pyplot as plt
import numpy as np
from Recognition import Record


size = [1000, 800]
CameraIndex = 0
pipe_numbers = ["91458","23964","96859","25890","12670"]


def ImgOnBoard(input):
    global size

    # Создаем черный фон (нулевой массив в формате BGR)
    black_background = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Определяем позицию, куда вставить изображение (центрируем)
    x_offset = (size[0] - input.shape[1]) // 2
    y_offset = (size[1] - input.shape[0]) // 2

    # Проверяем, чтобы изображение не выходило за границы фона
    if x_offset >= 0 and y_offset >= 0:
        # Размещаем изображение на черном фоне
        black_background[y_offset:y_offset + input.shape[0], x_offset:x_offset + input.shape[1]] = input
    else:
        print("Изображение слишком большое для фона!")

    return black_background


record = Record(r"files/settings v.2.json")

cap = cv2.VideoCapture(CameraIndex)
plt.ion()  # Режим интерактивного обновления

fig, ax = plt.subplots()
ax.axis('off')
img_display = ax.imshow(np.zeros((size[1], size[0], 3)))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем маркировку
        status, mess, text, marker, timer, previously, markers, logs, adjustments = record(frame, 1, pipe_numbers)
        # Выводим результат
        if status:
            print('--------------------------')

            # Результат работы
            print(text, " to ", marker)
            print('--------------------------')

            # Время выполнения
            print(timer, " sec.")
            print('--------------------------')

            # Отображение логов программы
            if logs != {} or adjustments != None:
                if logs != {}: print("Notes: ", logs)
                if adjustments != None: print(adjustments)
                print('--------------------------')

            # Вывод списка возможных вариантов
            if len(markers) > 1:
                print("List of options")
                for i in markers: print(i)
                print('--------------------------')

            # Итоговое изображение
            zone = cv2.cvtColor(mess, cv2.COLOR_GRAY2BGR)
            out = ImgOnBoard(zone)
        else:
            print("Error: ", mess)
            out = ImgOnBoard(previously)


        frame_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img_display.set_data(frame_rgb)
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Остановлено пользователем.")

cap.release()
plt.ioff()
