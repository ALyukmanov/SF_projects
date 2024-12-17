import numpy as np

def game_core_v3(number: int=1) -> int:
    """Функция угадывает загаданное число, уменьшая или увеличивая
    предположение в зависимости от того, больше или меньше оно нужного.
    
    Args:
        number (int): Загаданное число.

    Returns:
        int: Число попыток.
    """
    count = 0
    low = 1 # Устанавливаем нижнюю границу диапазона
    high = 100 # Устанавливаем верхнюю границу диапазона
    predict = np.random.randint(1, 101)
    predict = (low + high) // 2  # Начнем с среднего значения

    # Цикл продолжается, пока предсказанное число не совпадет с загаданным
    while number != predict:
        
        count += 1
        if number > predict:
            low = predict + 1  # Обновляем нижнюю границу
        elif number < predict:
            high = predict - 1  # Обновляем верхнюю границу
        predict = (low + high) // 2  # Вычисляем новое предположение как среднее значение

    return count



def score_game(game_core_v3) -> int:
    """За какое количество попыток в среднем за 10000 подходов угадывает наш алгоритм

    Args:
        game_core_v3 ([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    count_ls = []
    np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(10000))  # загадали список чисел

    for number in random_array:
        count_ls.append(game_core_v3(number)) 

    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за: {score} попытки")

if __name__ == '__main__':
    score_game(game_core_v3)