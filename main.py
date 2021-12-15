from MTree import MTree
from HaarWaveletTransform import HaarWaveletTransform
import matplotlib.pyplot as plt

def euclidean_distance(X, Y):
    # Calcular la distancia euclidea entre dos puntos
    return sum([(X[i] - Y[i]) ** 2 for i in range(len(Y))]) ** 0.5

# leer time_series.txt 
def open_file(filename):
    X = [] 
    with open(filename, 'r') as file:
        X = [] 
        for linea in file:
            numero = float(linea)
            X.append(numero)
    return X


#  poblar con los datos mediante un slide window
def sliding_window(X, Y, w, tree):
    data_set = []
    for i in range(len(X) - w+1):
        # print("Slide window: ", X[i:i+w])
        temp = X[i:i+w] 
        data_set.append(temp)
        tree.add(temp)
    resultado = list(tree.search(Y))
    print("Subsecuencia de coincidencia: ", resultado)
    return data_set, resultado


def plot_graph(t_x, t_y, X, Y):
    # plot  X and resultado
    plt.plot(t_x, X, 'b', label='Serie de Tiempo')

    for subsequence in resultado:
        plt.plot(t_y, subsequence, 'r', label='Subsecuencia')
        plt.plot(t_y, Y, 'g', label='Consulta')

    plt.legend()
    plt.show()


tree = MTree(euclidean_distance, max_node_size=5)

X = open_file('time_series.txt')
X = HaarWaveletTransform(X, 5)

# obtener la subsecuencia de consulta
Y = [float(y) for y in input("Ingresa la subsecuencia de consulta: ").split()]
Y = HaarWaveletTransform(Y, 5)
print("Transformada de Haar Wavelets(consulta): ", Y)
w = len(Y)

data_set, resultado = sliding_window(X, Y, w, tree)
t = 0
for i in range(len(data_set)):
    if data_set[i] in resultado:
        t = i
        break

t_y = [i+1 for i in range(t, t + w)]
t_x = [i for i in range(1, len(X)+1)]

plot_graph(t_x, t_y, X, Y)