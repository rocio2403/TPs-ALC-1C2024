#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    

    # suma_zieit = 0  
    # for i in range(n-1):
    #     z_i = np.array([[0]]*n)
    #     e_i = np.array([[0]]*n)
    #     e_i[i][0] = 1
    #     for j in range(i+1, n):
    #         pivote = Ac[i][i]
    #         z_ji = (Ac[j][i] / pivote)
    #         z_i[j] = z_ji
    #     print(i)
    #     print(z_i*(e_i.T)  )
    #     suma_zieit = suma_zieit + z_i*(e_i.T)    
            
    # I = np.eye(n)
    # L = I + suma_zieit

    # L = np.eye(n)

    # for col in range(n-1):
        
    #     for fila in range(col+1, n):
    #         pivote = Ac[col][col]
    #         a_ji = (Ac[fila][col] / pivote)
    #         L[fila][col] =  a_ji
            
    #         for k in range(col+1, n):
    #             Ac[fila][k] = Ac[fila][k] - a_ji*Ac[fila-1][k]
        
    #     print(Ac)
    
    L = np.eye(n)
    for k in range(n-1):
        pivote = Ac[k][k]
        print('--------------------')
        print(pivote)

        for f in range(k+1, n):
            print(Ac[f][k])
            coef = Ac[f][k] / pivote
            L[f][k] = coef
            for c in range(k, n):
                Ac[f][c] = Ac[f][c] + ((-coef)*Ac[k][c])

        print(Ac)
    ## hasta aqui
            
    U = np.triu(Ac)
    
    return L, U, cant_op

A = np.array([[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]])

L,u,cant_op = elim_gaussiana(A)
#%%

def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
    