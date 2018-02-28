class Hyperparams:
    # Path Info
    log_dir = './cycle_gan_vc_log'
    train_hdf5_path = './feat/trim_log_vctk.h5'
    infer_hdf5_path = './feat/trim_log_vctk.h5'
    
    # Data Loading Issues
    partition = 10

    # Networks
    lr = 0.0001
    batch_size = 1
    fix_seq_length = 128
    summary_period = 300
    save_period = 500
    LAMBDA_CYCLE = 10
    LAMBDA_IDENTITY = 5

    # Signal Processing
    sr = 16000
    n_fft = 1024 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 500 # Number of inversion iterations
