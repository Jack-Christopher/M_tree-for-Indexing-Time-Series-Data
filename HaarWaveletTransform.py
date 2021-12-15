def HaarWaveletTransform ( signal, level ):
    s = .5                  # ampliacion (scaling)

    h = [ 1,  1 ]           # filtro lowpass 
    g = [ 1, -1 ]           # filtro highpass
    f = len ( h )           # longitud del filtro

    t = signal;              # arreglo de trabajo
    l = len ( t )           # longitud de la señal actual
    y = [0] * l             # se inicializa el arreglo de salida

    t = t + [ 0, 0 ]

    for i in range ( level ):
        y [ 0:l ] = [0] * l; # inicializa para el siguiente nivel
        l2 = l // 2;         

        for j in range ( l2 ):            
            for k in range ( f ):                
                y [j]    += t [ 2*j + k ] * h [ k ] * s
                y [j+l2] += t [ 2*j + k ] * g [ k ] * s

        l = l2;              # se continua con la aproximacion del siguiente nivel
        t [ 0:l ] = y [ 0:l ] 

    return y
