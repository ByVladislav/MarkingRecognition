from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2, torch, re, time, easyocr, os, random, json
from difflib import SequenceMatcher
from ultralytics import YOLO
from PIL import Image
import numpy as np


# Класс для нахождения и расшифровки маркировки
class Record:

    # Инициализатор
    def __init__(self, path = r'files/settings.json'):

        # Модели нейросетей
        self.modelDetect = None
        self.processor = None
        self.modelRecordTrOCR = None

        # Загружаем модель Easy OCE
        self.modelRecordEasy_ocr = easyocr.Reader(['ru'])

        # Загружаем настройки
        self.settings = None
        self.SetSettings(path)
        
        # Определяем устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelRecordTrOCR.to(self.device)
        print("Using device:", self.device)

        # Список для хранения логов
        self.finding = list()

    # Функция для установки настроек из файла
    def SetSettings(self, path):
        # Подгружаем настройки из JSON файла
        try:
            with open(path) as f:
                self.settings = json.load(f)
        except:
            self.settings = {"top_percent": 0.45, "bottom_percent": 0.35, "height": 75, "Significance": 0.5,
                             "BestScore": 0.6, "path_YOLO": "files/model v1.pt", "path_TROCR": "files/trocr-base-ru"}

        # Подключаем модель поиска рамки
        self.modelDetect = YOLO(self.settings["path_YOLO"])

        # Подключаем модель распознавания текста
        self.processor = TrOCRProcessor.from_pretrained(self.settings["path_TROCR"])
        self.modelRecordTrOCR = VisionEncoderDecoderModel.from_pretrained(self.settings["path_TROCR"])

    # Функция обработчик
    def __call__(self, input, numModel, pipe_numbers=None,
                 structure={"company": "ТМК", "factory": "ЧТПЗ", "steel": "30Г2", "number": 5}):

        # Запоминаем время начала обработки
        TimePoint = time.time()

        # Оброезаем картинку
        gray, color = self.CropArea(input)

        # Ищем и подготавливаем маркировку
        status, zone = self.MarkerDetectYOLO(gray, color)
        if status == False: return False, zone, None, None, None, color, None, None, None

        # Распознаём маркировку
        if numModel == 0:
            status, text = self.MarkerRecordTrOCR(zone)
            if status == False: return False, text, None, None, None, color, None, None, None
        elif numModel == 1:
            text = self.MarkerRecordEasy_ocr(zone)

        # Структурируем текст
        status, err, marker, numbers, logs = self.StrictText(text, structure, pipe_numbers)
        if status == False: return False, err, None, None, None, color, None, None, None

        return True, zone, text, marker, round(time.time() - TimePoint, 3), color, numbers, logs, err

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
        new_width = int((self.settings["height"] / h) * w)

        Part8 = cv2.resize(Part7, (new_width, self.settings["height"]))

        # Инвертируем
        Part9 = cv2.bitwise_not(Part8)

        return True, Part9

    # Расшифровка маркировки base_ocr
    def MarkerRecordTrOCR(self, input):

        # Переводим картинку в формат RGB
        Part1 = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        Part2 = Image.fromarray(Part1)

        # Подготавливаем картинку к обработке
        Part3 = self.processor(images=Part2, return_tensors="pt").pixel_values.to(self.device)

        # Распознаём текст
        with torch.no_grad():
            generated_ids = self.modelRecordTrOCR.generate(
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
        if not os.path.isdir("files"): os.mkdir("files")
        if not os.path.isdir("files/temp"): os.mkdir("files/temp")
        name = f"files/temp/temp_{random.randint(10000,99999)}.jpg"

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
        top_crop = int(height * self.settings["top_percent"])
        bottom_crop = int(height * self.settings["bottom_percent"])

        Part3 = Part2[top_crop:height - bottom_crop, 0:width]
        Part4 = input[top_crop:height - bottom_crop, 0:width]

        return Part3, Part4

    # Структурирование текста
    def StrictText(self, text, structure, pipe_numbers=None):

        self.finding = list()

        # Удаляем все пробелы из входного текста для унификации
        original_text = re.sub(r'\s+', '', text)
        clean_text = original_text.upper()  # Сохраняем оригинал для логов

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
        pipe_numbers = self._find_pipe_number(clean_text, structure["number"], pipe_numbers)
        if pipe_numbers == None:
            return False, {"error_code": "002", "error": "Unable to find pipe number"}, None, None, None

        # Формируем сообщение об изменениях
        mess = {"error_code": "003", "error": self.finding} if self.finding else {}

        # Определяем лучший номер
        pipe_number = self._select_best_pipe_number(pipe_numbers, structure["number"], clean_text)
        if pipe_number == None:
            corrected = f"{structure['company']} {structure['factory']} {valid_steel} {pipe_numbers[0]}"
            return True, {"error_code": "004", "error": "Unable to accurately determine the room number"}, corrected, pipe_numbers, mess

        # Собираем корректную маркировку
        corrected = f"{structure['company']} {structure['factory']} {valid_steel} {pipe_number}"


        return True, None, corrected, pipe_numbers, mess

    # Ищет SIZE-значный номер трубы
    def _find_pipe_number(self, text, size, valid_numbers=None):

        if valid_numbers == None:
            # Сначала ищем в конце строки
            end_match = re.search(r'(\d{{{size}}})\D*$'.format(size=size), text)
            if end_match:
                return [(end_match.group(1), 1.0)]

            # Если не нашли в конце, ищем в любом месте
            any_match = re.search(r'(\d{{{size}}})'.format(size=size), text)
            if any_match:
                return [(any_match.group(1), 1.0)]

            return None

        # Извлекаем все последовательности цифр длиной от 3 до SIZE символов
        digit_sequences = re.findall(rf'\d{{3,{size}}}', text)

        if not digit_sequences or not valid_numbers:
            return None

        matches = []

        # Для каждого допустимого номера ищем лучшее частичное совпадение
        for valid_num in valid_numbers:
            best_score = 0.0

            for seq in digit_sequences:
                # Для полного совпадения
                if len(seq) == size:
                    similarity = SequenceMatcher(None, seq, valid_num).ratio()
                    if similarity > best_score:
                        best_score = similarity

                # Для частичного совпадения (3- и более цифр)
                elif 3 <= len(seq) < size:
                    partial_score = self._find_partial_matches(seq, valid_num)
                    # Нормализуем оценку (частичное совпадение не может быть 1.0)
                    normalized_score = partial_score * (len(seq) / size) * 0.9
                    if normalized_score > best_score:
                        best_score = normalized_score

            if best_score > self.settings["Significance"]:  # Порог значимости
                matches.append((valid_num, best_score))

        # Сортируем по убыванию оценки
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches if matches else None

    # Выбор лучшего варианта
    def _select_best_pipe_number(self, matches, size, text):

        if not matches:
            return None

        # Извлекаем все последовательности цифр длиной от 3 до SIZE символов
        digit_sequences = re.findall(rf'\d{{3,{size}}}', text)

        best_num, best_score = matches[0]

        # Проверяем, что лучший результат значительно лучше альтернатив
        if len(matches) > 1:
            next_score = matches[1][1]
            confidence = best_score - next_score

            # Если разница менее 20%, считаем результат неоднозначным
            if confidence < 0.2:
                options = [f"{num}({score:.2f})" for num, score in matches[:3]]
                self._log_addition("pipe_number_warning",
                                   f"Low confidence ({confidence:.2f}): {', '.join(options)}")

        # Логируем исправление, если оценка < 0.95
        if best_score < 0.95:
            source = next((seq for seq in digit_sequences
                           if SequenceMatcher(None, seq, best_num).ratio() > 0.5), "unknown")
            self._log_correction("pipe_number",
                                 f"partial '{source}' (score: {best_score:.2f})",
                                 f"corrected to '{best_num}'")

        return best_num if best_score > self.settings["BestScore"] else None  # Минимальный порог

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

    # Поиск в списке похожих номеров
    def _find_partial_matches(self, recognized, full_number):

        if len(recognized) < 3:  # Слишком короткий фрагмент
            return 0.0

        max_similarity = 0.0

        # Проверяем все возможные позиции во всем номере
        for i in range(len(full_number) - len(recognized) + 1):
            fragment = full_number[i:i + len(recognized)]
            similarity = SequenceMatcher(None, recognized, fragment).ratio()
            if similarity > max_similarity:
                max_similarity = similarity

        return max_similarity

    # Логирует добавление отсутствующей части
    def _log_addition(self, part_name, added):
        self.finding.append(f"Adding {part_name}: {added}")

    # Логирует исправление части текста
    def _log_correction(self, part_name, original, corrected):

        if part_name == "steel":
            # Для стали берем только марку без номера
            steel_only = self._extract_steel_part(original)
            self.finding.append(f"Correction {part_name}: {steel_only}->{corrected}")
        else:
            self.finding.append(f"Correction {part_name}: {original}->{corrected}")