from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2, torch, re, time, easyocr, os, random
from difflib import SequenceMatcher
from ultralytics import YOLO
from PIL import Image
import numpy as np


# Класс для нахождения и расшифровки маркировки
class Record:
    # Инициализатор
    def __init__(self, top, bottom, pathModel=None, pathOCR=None):
        # Подключаем модель поиска рамки
        if pathModel is not None:
            self.model = YOLO(pathModel)
        else:
            self.modelDetect = YOLO(r"files/model v2.pt")

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
        self.top_percent = top
        self.bottom_percent = bottom
        self.height = 75

        self.finding = list()


    # Поиск маркировки и выделение зоны интереса
    def MarkerDetectYOLO(self, gray, color):

        # Находим маркировку на трубе
        results = self.modelDetect(color)

        if len(results[0]) == 0: return False, {"error_code":"001", "error":"Not detecting marking"}

        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
        Part1 = gray[y1:y2, x1:x2]

        Part2 = cv2.rotate(Part1, cv2.ROTATE_180)

        # Удаляем шум
        kernel = np.ones((1, 1), np.uint8)
        Part3 = cv2.morphologyEx(Part2, cv2.MORPH_OPEN, kernel)

        # Убираем тени
        background = cv2.medianBlur(Part3, 21)
        Part4 = cv2.addWeighted(Part3, 1, background, -0.7, 0)

        # Проводим бинаризацию
        _, Part5 = cv2.threshold(Part4, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Размываем, чтобы сгладить бинаризацию
        Part6 = cv2.blur(Part5, (5, 5))

        # Уменьшаем яркость
        Part7 = cv2.subtract(Part6, 40)

        # Уменьшаем картинку для сглаживания бинаризации
        h, w = Part7.shape[:2]
        new_width = int((self.height / h) * w)

        Part8 = cv2.resize(Part7, (new_width, self.height))

        # Инвертируем
        Part9 = cv2.bitwise_not(Part8)

        return True, Part9

    # Расшифровка маркировки base_ocr
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

        return True, text

    # Расшифровка маркировки easy_OCR
    def MarkerRecordEasy_ocr(self, input):

        # Создаем имя временного файла
        name = f"temp_{random.randint(10000,99999)}.jpg"

        # Сохраняем временый файл
        cv2.imwrite(name, input)

        # Распознаем текст
        result = self.modelRecordEasy_ocr.readtext(name, detail=0)

        # Объединяем результаты распознования
        text=""
        for i in result: text += i

        # Удаляем временый файл
        os.remove(name)

        return text

    def CropArea(self, input):

        # Переводим в серый и размываем
        Part1 = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        Part2 = cv2.GaussianBlur(Part1, (3, 3), 0)

        # Обрезаем по зоне интереса
        height, width = Part2.shape[:2]
        top_crop = int(height * self.top_percent)
        bottom_crop = int(height * self.bottom_percent)

        Part3 = Part2[top_crop:height - bottom_crop, 0:width]
        Part4 = input[top_crop:height - bottom_crop, 0:width]

        return Part3, Part4

    # Структурирование текста
    def StrictText(self, text, structure):
        self.finding = list()

        # Удаляем все пробелы из входного текста для унификации
        clean_text = re.sub(r'\s+', '', text)
        original_text = clean_text.upper()  # Сохраняем оригинал для логов

        # Проверяем наличие обязательных частей
        required_parts = [
            ("company", structure["company"]),
            ("factory", structure["factory"])
        ]

        # Проверяем обязательные части
        for part_name, part in required_parts:
            if part not in clean_text:
                matches = self._find_similar_substring(clean_text, part)
                if matches:
                    best_match = max(matches, key=lambda x: x[1])
                    if best_match[0] != part:
                        self._log_correction(part_name, best_match[0], part)
                        clean_text = clean_text.replace(best_match[0], part)
                else:
                    self._log_addition(part_name, part)
                    clean_text = part + clean_text

        # Ищем марку стали
        steel_pattern = re.compile(rf'{re.escape(structure["factory"])}(.{{2,10}})')
        steel_match = steel_pattern.search(clean_text)

        if steel_match:
            steel_part = steel_match.group(1)
            cleaned_steel = re.sub(r'[^a-zA-Zа-яА-Я0-9-]', '', steel_part)
            valid_steel = self._validate_steel_grade(cleaned_steel, structure["steel"])

            if valid_steel != cleaned_steel:
                self._log_correction("steel", cleaned_steel, valid_steel)
        else:
            valid_steel = structure["steel"]
            self._log_addition("steel", valid_steel)

        # Ищем номер трубы
        pipe_number = self._find_pipe_number(clean_text, structure["number"])
        if pipe_number == None:
            return False, {"error_code": "002", "error": "Unable to find pipe number"}, None

        # Собираем корректную маркировку
        corrected = f"{structure['company']} {structure['factory']} {valid_steel} {pipe_number}"

        # Формируем сообщение об изменениях
        mess = {"error_code": "003", "error": self.finding} if self.finding else {}

        return True, mess, corrected

    # Логирует добавление отсутствующей части
    def _log_addition(self, part_name, added):
        self.finding.append(f"Adding {part_name}: {added}")

    # Находит подстроки в тексте, похожие на target
    def _find_similar_substring(self, text, target):
        matches = []
        target_len = len(target)

        for i in range(len(text) - target_len + 1):
            substring = text[i:i + target_len]
            similarity = SequenceMatcher(None, substring, target).ratio()
            if similarity > 0.6:
                matches.append((substring, similarity))

        return matches

    # Проверяет и корректирует марку стали
    def _validate_steel_grade(self, steel_text, default_grade):
        """Проверяет и корректирует марку стали, отделяя ее от номера трубы"""
        # Удаляем все пробелы и недопустимые символы
        steel_text = re.sub(r'[^a-zA-Zа-яА-Я0-9-]', '', steel_text)

        # Отделяем марку стали от возможного номера трубы
        steel_part = self._extract_steel_part(steel_text)

        valid_grades = [
            '30Г2', '20Mn5', '20MnB4', '20MnCr5',
            '20MnCrS5', '20MnMoNi4-5', '36NiCrMo16'
        ]

        # Проверяем точное соответствие
        if steel_part in valid_grades:
            return steel_part

        # Ищем похожие марки
        for grade in valid_grades:
            if SequenceMatcher(None, steel_part, grade).ratio() > 0.7:
                return grade

        return default_grade

    # Извлекает часть текста, которая может быть маркой стали
    def _extract_steel_part(self, text):

        # Паттерны для разных типов марок стали
        patterns = [
            r'^\d{2}[A-Za-z]+[A-Za-z0-9-]+',  # Для 20Mn5, 20MnB4 и т.д.
            r'^\d{2}[А-Яа-я]+\d*',  # Для 30Г2 и подобных
            r'^\d{2}[A-Za-z]+[A-Za-z0-9-]+',  # Для 36NiCrMo16
        ]

        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                return match.group()

        # Если ни один паттерн не подошел, возвращаем первые 4 символа (как эвристику)
        return text[:4]

    # Логирует исправление части текста
    def _log_correction(self, part_name, original, corrected):
        if part_name == "steel":
            # Для стали берем только марку без номера
            steel_only = self._extract_steel_part(original)
            self.finding.append(f"Correction {part_name}: {steel_only}->{corrected}")
        else:
            self.finding.append(f"Correction {part_name}: {original}->{corrected}")

    # Ищет 5-значный номер трубы
    def _find_pipe_number(self, text, size):
        # Сначала ищем в конце строки
        end_match = re.search(r'(\d{{{size}}})\D*$'.format(size=size), text)
        if end_match:
            return end_match.group(1)

        # Если не нашли в конце, ищем в любом месте
        any_match = re.search(r'(\d{{{size}}})'.format(size=size), text)
        if any_match:
            return any_match.group(1)

        return None

    # Функция обработчик
    def __call__(self, input, numModel, structure={"company": "ТМК", "factory": "ЧТПЗ", "steel": "30Г2", "number": 5}):

        # Запоминаем время начала обработки
        TimePoint = time.time()

        # Оброезаем картинку
        gray, color = self.CropArea(input)

        # Ищем и подготавливаем маркировку
        status, zone = self.MarkerDetectYOLO(gray, color)
        if status == False: return False, zone, None, None, None, color, None

        # Распознаём маркировку
        if numModel == 0:
            status, text = self.MarkerRecordBase_ru(zone)
            if status == False: return False, text, None, None, None, color, None
        elif numModel == 1:
            text = self.MarkerRecordEasy_ocr(zone)

        # Структурируем текст
        status, mess, marker = self.StrictText(text, structure)
        if status == False: return False, mess, None, None, None, color, None

        return True, zone, text, marker, round(time.time()-TimePoint, 3), color, mess