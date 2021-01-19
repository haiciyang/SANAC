
  
import numpy as np
import math

SAMPLE_RATE = 16000

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)

# generate Mel filter bank
def melFilterBank(numCoeffs, fftSize = None):
    minHz = 0
    maxHz = SAMPLE_RATE / 2            # max Hz by Nyquist theorem
    if (fftSize is None):
        numFFTBins = WINDOW_SIZE
    else:
        numFFTBins = fftSize // 2 + 1

    maxMel = freqToMel(maxHz)
    minMel = freqToMel(minHz)

    # we need (numCoeffs + 2) points to create (numCoeffs) filterbanks
    melRange = np.array(range(numCoeffs + 2))
    melRange = melRange.astype(np.float32)
    
#     print(melRange)

    # create (numCoeffs + 2) points evenly spaced between minMel and maxMel
    melCenterFilters = melRange * (maxMel - minMel) / (numCoeffs + 1) + minMel
#     print(melCenterFilters)

    for i in range(numCoeffs + 2):
        # mel domain => frequency domain
        melCenterFilters[i] = melToFreq(melCenterFilters[i])

        # frequency domain => FFT bins
        melCenterFilters[i] = int(math.floor(numFFTBins * melCenterFilters[i] / maxHz))
#         print(melCenterFilters[i])x
    
    # create matrix of filters (one row is one filter)
    filterMat = np.zeros((numCoeffs, int(numFFTBins)))

    # generate triangular filters (in frequency domain)
    for i in range(1, numCoeffs + 1):
        filter = np.zeros(numFFTBins)
        
        startRange = int(melCenterFilters[i - 1])
        midRange   = int(melCenterFilters[i])
        endRange   = int(melCenterFilters[i + 1])
        
#         print(startRange, midRange, endRange)
        
        for j in range(startRange, midRange):
            filter[j] = (float(j) - startRange) / (midRange - startRange)
        for j in range(midRange, endRange):
            filter[j] = 1 - ((float(j) - midRange) / (endRange - midRange))
        
        filterMat[i - 1] = filter

    # return filterbank as matrix
    return filterMat

