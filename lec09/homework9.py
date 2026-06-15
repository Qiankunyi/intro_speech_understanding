import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)
    
    num_frames = int((len(waveform) - frame_length) / step) + 1

    energies = np.zeros(num_frames)
    for t in range(num_frames):
        frame = waveform[t * step : t * step + frame_length]
        energies[t] = np.sum(frame ** 2)

    threshold = 0.10 * np.max(energies)

    segments = []
    in_speech = False
    start_sample = 0
    
    for t in range(num_frames):
        if energies[t] > threshold and not in_speech:
            in_speech = True
            start_sample = t * step
        elif energies[t] <= threshold and in_speech:
            in_speech = False
            end_sample = t * step + frame_length
            segments.append(waveform[start_sample:end_sample])

    if in_speech:
        end_sample = (num_frames - 1) * step + frame_length
        segments.append(waveform[start_sample:end_sample])
        
    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []
    
    N = int(0.004 * Fs)
    step = int(0.002 * Fs)
    half_N = int(N / 2)  
    
    for segment in segments:
        pre_emphasized = segment[1:] - 0.97 * segment[:-1]

        num_frames = int((len(pre_emphasized) - N) / step) + 1

        log_spectra = []
        for t in range(num_frames):
            frame = pre_emphasized[t * step : t * step + N]

            spectrum = np.abs(np.fft.fft(frame))[:half_N]

            log_spectrum = np.log(spectrum + 1e-10)
            log_spectra.append(log_spectrum)

        average_log_spectrum = np.mean(log_spectra, axis=0)
        models.append(average_log_spectrum)
        
    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)
    
    Y = len(models)      
    K = len(test_models)  
    
    sims = np.zeros((Y, K))
    test_outputs = []
    
    for y in range(Y):
        model_vec = models[y]
        model_norm = np.linalg.norm(model_vec)
        
        for k in range(K):
            test_vec = test_models[k]
            test_norm = np.linalg.norm(test_vec)
            
            if model_norm > 0 and test_norm > 0:
                
                sims[y, k] = np.dot(model_vec, test_vec) / (model_norm * test_norm)
            else:
                sims[y, k] = 0.0
                
    for k in range(K):
        best_model_idx = np.argmax(sims[:, k])
        test_outputs.append(labels[best_model_idx])
        
    return sims, test_outputs


