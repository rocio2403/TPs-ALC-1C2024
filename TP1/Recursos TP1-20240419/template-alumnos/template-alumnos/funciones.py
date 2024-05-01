import numpy as np
import networkx as nx
from scipy.linalg import solve_triangular


def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
    	line = f.readline()
    	i = int(line.split()[0]) - 1
    	j = int(line.split()[1]) - 1
    	W[j,i] = 1.0
    f.close()
    
    return W

def dibujarGrafo(W, print_ejes=True):
    
    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}
    
    N = W.shape[0]
    G = nx.DiGraph(W.T)
    
    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])
    
    nx.draw(G, pos=nx.spring_layout(G), **options)
    
#%%
def factorizacion_LU(A):
    m=A.shape[0]  #Numero de filas
    n=A.shape[1]   #Numero de columnas
    matriz = A.copy() #Para no modificar la matriz original
    
    if m!=n:
        print('Matriz no cuadrada, no es posible factorizar')
        
    else:
        L = np.eye(n) #inicializamos L
        for k in range(n-1):
            pivote = matriz[k][k]    #Tomamos un pivote
            # print('--------------------')
            # print(pivote)
    
            for f in range(k+1, n):   #Iteramos sobre esa columna
                coef = matriz[f][k] / pivote
                L[f][k] = coef         #Guardamos coeficiente
                for c in range(k, n):
                    matriz[f][c] = matriz[f][c] + ((-coef)*matriz[k][c]) #Actualizamos los demas nros de la matriz
                    
        U = matriz #Guardamos como u, la matriz triangulada
    return L, U

def calcular_grado(matriz,i):
    res =  np.sum(matriz[:, i])
    return res

def componentes_Pagerank(W, p):
    """
    Dada una matriz de conectividad W y una probabilidad p,
    calcula las componentes de la ecuación 6.
    """
    n = W.shape[0] #Cantidad de páginas
    #creamos e
    e = np.ones((n,1)) #Vector columna de unos
    #creamos D (matriz diagonal con djj = 1/cj o 0 ) 
    #para eso calculamos cj (grado de la página j: cantidad de links salientes de j)
    D = np.eye(n)
    for i in range(len(D)):
        if calcular_grado(W, i) == 0:
           D[i][i]=0
        else:
          D[i][i] = 1 / calcular_grado(W, i)  #el grado es la suma sobre la columna
        
    #creamos z y luego trasponemos
    
    z = np.zeros((n,1)) #Vector columna  
    for j in range(n):
        if calcular_grado(W, j) == 0:
            z[j] = 1/n
        else:
            z[j] = (1-p)/n
    #ahora trasponemos z
    z_t =  np.transpose(z)
    return p, W, e, D, z_t


#%%
def calcularPuntajes(w,p):
    p, W, e, D, z_t = componentes_Pagerank(w, p)
    #resolvemos la ecuacion 6
    #para eso podemos armar la matriz con los datos obtenidos
    #Luego hacer factorizacion LU
    #resolver el sistema con L y = b , U X = y
    #así obtenemos x, ya que asumimos que gamma(?es 1
    #por lo que nuestro sistema queda :
    #  (I - PWD)X = e
    #Por lo cual buscamos la matriz I-PWD
    identidad = np.eye(W.shape[0])
    
    A = p*(W@D) + (e@z_t)
    M = identidad - p*(W@D)
    L, U = factorizacion_LU(M)
    
    #resolvemos sistema ya que son matrices triangulares
    y = solve_triangular(L, e, lower=True)
    x = solve_triangular(U, y, lower=False)

    #normalizamos x
    x = x/np.sum(x)

    return x

#%%

def calcularRanking(M, p):
    npages = M.shape[0]
    rnk = np.arange(0, npages) # ind[k] = i, la pagina k tienen el iesimo orden en la lista.
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha 
    # Codigo
    scr = calcularPuntajes(M, p).reshape(12,)
    pagina_score = list(zip(range(npages), scr))
    pagina_score.sort(key=lambda x: x[1], reverse=True)
    for t in range(npages):
        pagina = pagina_score[t][0]
        rnk[pagina] = t+1        
    return rnk, scr 

def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)
    
    return output

    
    
#ARCHIVOS DE ENTRADA
archivo_test = './tests/test_dosestrellas.txt'
#CARGA DE ARCHIVO EN GRAFO
W = leer_archivo(archivo_test)

dibujarGrafo(W, print_ejes=False)

# defino la probabilidad de salto de continuar los links de la pagina actual
p = 0.5


# Realizo el test unitario para el calculo del mayor score, que pruebe que el codigo funciona correctamente.
print('*'*50)
print('Test unitario 1')
try:
    assert(np.isclose(obtenerMaximoRankingScore(W, p), 0.1811))
except:
    print('OUCH!! - No paso el test unitario')
else:
    print('BIEN! - Paso correctamente el test unitario')
print('*'*50)
    
    
    