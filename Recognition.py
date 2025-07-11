from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2, torch, re, time, easyocr, os, random
from ultralytics import YOLO
from PIL import Image
import numpy as np


# Класс для нахождения и расшифровки маркировки
class Record:
    # Инициализатор
    def __init__(self, pathModel=None, pathOCR=None):
        # Подключаем модель поиска рамки
        if pathModel is not None:
            self.model = YOLO(pathModel)
        else:
            self.modelDetect = YOLO(r"files/model.pt")

        # Подключаем модель распознавания текста
        if pathOCR is not None:
            self.processor = TrOCRProcessor.from_pretrained(pathOCR)
            self.modelRecordBase_ru = VisionEncoderDecoderModel.from_pretrained(pathOCR)
        else:
            self.processor = TrOCRProcessor.from_pretrained(r"files/trocr-base-ru")
            self.modelRecordBase_ru = VisionEncoderDecoderModel.from_pretrained(r"files/trocr-base-ru")

        # Определяем устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.modelRecordBase_ru.to(self.device)

        self.modelRecordEasy_ocr = easyocr.Reader(['ru'])

        # Настройка параметров
        self.top_percent = 0.45
        self.bottom_percent = 0.35
        self.height = 75


    # Поиск маркировки и выделение зоны интереса
    def MarkerDetect(self, input):

        # Переводим в серый и размываем
        Part1 = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        Part2 = cv2.GaussianBlur(Part1, (3, 3), 0)

        # Обрезаем по зоне интереса
        height, width = Part2.shape[:2]
        top_crop = int(height * self.top_percent)
        bottom_crop = int(height * self.bottom_percent)

        Part3 = Part2[top_crop:height - bottom_crop, 0:width]
        Part4 = input[top_crop:height - bottom_crop, 0:width]

        # Находим маркировку на трубе
        results = self.modelDetect(Part4)

        if len(results[0]) == 0: return False, {"error_code":"001", "error":"Not detecting marking"}

        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
        Part5 = Part3[y1:y2, x1:x2]

        Part6 = cv2.rotate(Part5, cv2.ROTATE_180)

        # Удаляем шум
        kernel = np.ones((1, 1), np.uint8)
        Part7 = cv2.morphologyEx(Part6, cv2.MORPH_OPEN, kernel)

        # Убираем тени
        background = cv2.medianBlur(Part7, 21)
        Part8 = cv2.addWeighted(Part7, 1, background, -0.7, 0)

        # Проводим бинаризацию
        _, Part9 = cv2.threshold(Part8, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Размываем, чтобы сгладить бинаризацию
        Part10 = cv2.blur(Part9, (5, 5))

        # Уменьшаем яркость
        Part11 = cv2.subtract(Part10, 40)

        # Уменьшаем картинку для сглаживания бинаризации
        h, w = Part11.shape[:2]
        new_width = int((self.height / h) * w)

        Part12 = cv2.resize(Part11, (new_width, self.height))

        # Инвертируем
        Part13 = cv2.bitwise_not(Part12)

        return True, Part13

    # Расшифровка маркировки
    def MarkerRecordBase_ru(self, input):
        # Переводим картинку в формат RGB
        Part1 = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        Part2 = Image.fromarray(Part1)

        # Подготавливаем картинку к обработке
        Part3 = self.processor(images=Part2, return_tensors="pt").pixel_values.to(self.device)

        # Распознаём текст
        with torch.no_grad():
            generated_ids = self.modelRecordBase_ru.generate(
                Part3,
                max_length=128,
                num_beams=8,
                early_stopping=True
            )
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Структурируем полученый текст
        marker = self.StrictText(text)

        return True, {"text": text, "marker": marker}


    # Структурирование текста
    def StrictText(self, text):

        # Маска для структурирования
        text = re.sub(r"[^A-Za-zА-Яа-я0-9]", "", text).upper()


        # Функция струтурирования
        def stricting(text, check, count, offset):
            result, i = '', offset
            while i < len(text) and len(result) < count:
                if check(text[i]):
                    result += text[i]
                i += 1
            return result, i

        # Структурируем
        pos = 0
        part1, pos = stricting(text, str.isalpha, 3, pos)  # Производитель
        part2, pos = stricting(text, str.isalpha, 4, pos)  # Завод
        part3, pos = stricting(text, str.isdigit, 2, pos)  # Процент углерода
        part4, pos = stricting(text, str.isalpha, 1, pos)  # Добавочный металл
        part5, pos = stricting(text, str.isdigit, 1, pos)  # Процентное содержание добавочного металла
        part6, pos = stricting(text, str.isdigit, 5, pos)  # Номер

        return f"{part1} {part2} {part3}{part4}{part5} {part6}"


    def MarkerRecordEasy_ocr(self, input):

        name = f"temp_{random.randint(10000,99999)}.jpg"

        cv2.imwrite(name, input)

        result = self.modelRecordEasy_ocr.readtext(name, detail=0)

        os.remove(name)

        return {"text": result, "marker": result}

    # Функция обработчик
    def __call__(self, input, numModel):

        # Запоминаем время начала обработки
        TimePoint = time.time()

        # Ищем и подготавливаем маркировку
        status, zone = self.MarkerDetect(input)
        if status == False: return False, zone, None, None, None

        # Распознаём маркировку
        if numModel == 0:
            status, data = self.MarkerRecordBase_ru(zone)
            if status == False: return False, data, None, None, None
        elif numModel == 1:
            data = self.MarkerRecordEasy_ocr(zone)

        return True, zone, data["text"], data["marker"], round(time.time()-TimePoint, 3)
