import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    num_s = int(0.5 * Fs)

    n = np.arange(num_s)

    f1 = f                          
    f2 = f * np.power(2, 4/12)     
    f3 = f * np.power(2, 7/12)

    x1 = np.cos(2 * np.pi * f1 * n / Fs)
    x2 = np.cos(2 * np.pi * f2 * n / Fs)
    x3 = np.cos(2 * np.pi * f3 * n / Fs)

    x = x1 + x2 + x3

    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    W = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for n in range(N):
            
            W[k, n] = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)
            
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    N = len(x)

    X = np.fft.fft(x)

    X_m = np.abs(X)

    half_N = N // 2
    pos_m = X_m[:half_N]

    top_3_k = np.argsort(pos_m)[-3:][::-1]

    fre = []
    for k in top_3_k:
        f = (k / N) * Fs
        fre.append(f)
    
    fre_sorted = sorted(fre)

    f1 = fre_sorted[0]
    f2 = fre_sorted[1]
    f3 = fre_sorted[2]

    return f1, f2, f3
