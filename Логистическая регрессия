from matplotlib import pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation

#создаем точки
data = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[5,2],[2,3],[4,3]])
#истинная принадлежность к какому либо классу
metki = np.array([0,0,0,0,1,1,1,1])

#сигмоида дающая значение на выходе от 0 до 1
def sigmoid(x):
    res=1/(1+np.exp((-1)*x))
    return res

#вычисление оценки
def score(vesa: np.ndarray, bias: float, tochka: np.ndarray) -> float:
    return np.dot(vesa, tochka) + bias

#Предсказание принадлежности на основании вычисленной оценки через сигмоиду
def prediction(vesa, bias, tochka):
    return sigmoid(score(vesa, bias, tochka))

#логистическая потеря оценки относительно настоящего значения
def log_loss(vesa, bias, tochka, metka):
    pred = prediction(vesa, bias, tochka)
    return -metka*np.log(pred) - (1-metka)*np.log(1-pred)

#вычисление тотальной ошибки всего списка точек
def total_log_loss(vesa, bias, tochki, metka):
    total_error = 0
    for i in range(len(tochki)):
        total_error += log_loss(vesa, bias,tochki[i], metka[i])
    return total_error

#изминение весов в зависимости от разности с истинной меткой
def logistic_step(vesa, bias, tochka,metka, learning_rate = 0.01):
     pred = prediction(vesa, bias, tochka)
     for i in range(len(vesa)):
         vesa[i] += (metka-pred)*tochka[i]*learning_rate
     bias += (metka-pred)*learning_rate
     return vesa, bias

#запуск по всем точкам
# сначала задаем нулевой биас и единичные веса
def logistic_regression_algorithm(tochki, metki, learning_rate = 0.01,cycles = 1000):
     #plt.utils.plot_points(tochki, metki)
     vesa = [1.0 for i in range(len(tochki[0]))]
     bias = 0.0
     errors = []
     for i in range(cycles):
         errors.append(total_log_loss(vesa, bias,tochki, metki))
         j = random.randint(0, len(tochki)-1)
         vesa, bias = logistic_step(vesa, bias,tochki[j],metki[j])
         update_plot(vesa, bias, tochki, metki)
         

     return vesa, bias

def update_plot(vesa, bias, tochki, metki):
    plt.clf()

    # Рисуем точки
    for i in range(len(tochki)):
        if metki[i] == 0:
            plt.scatter(tochki[i][0], tochki[i][1], color="blue")
        else:
            plt.scatter(tochki[i][0], tochki[i][1], color="green")

    # Рисуем линию
    x = np.linspace(-1, 6, 100)
    y = -(vesa[0] * x + bias) / vesa[1]

    plt.plot(x, y, color="red")

    plt.xlim(-1, 6)
    plt.ylim(-1, 5)
    plt.grid()

    plt.pause(0.03)


print(logistic_regression_algorithm(data, metki))


