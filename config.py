class Config:
    NB_CLASSES      = 10
    RAW_DATA        = '/storage/andn/data/music/train'
    RAW_TEST_DATA   = '/storage/andn/data/music/test'
    DATA_WAV        = '/storage/andn/data/music/data_wav'
    DATA_MP3        = '/storage/andn/data/music/data_mp3'

    TRAIN_DATA_MP3  = '/storage/andn/data/music/data_mp3/train'
    TEST_DATA_MP3   = '/storage/andn/data/music/data_mp3/test'

    TRAIN_DATA_WAV  = '/storage/andn/data/music/data_wav/train'
    TEST_DATA_WAV   = '/storage/andn/data/music/data_wav/test'

    TRAIN_CSV       = '/storage/andn/data/music/train.csv'
    TEST_CSV        = '/storage/andn/data/music/test.csv'

    NPY_SLICES_PATH = '/storage/andn/data/music/npy_slices'


    # local machine
    NAME_LOCAL_MACHINE = 'mindu'

    # audio
    NB_CLASS        = 2

    # spectrogram and silence
    FFT_SIZE        = 512  # window size for the FFT
    STEP_SIZE       = int(FFT_SIZE / 4)  # distance to slide along the window (in time)
    LOWCUT          = 500  # Hz # Low cut for our butter bandpass filter
    HIGHTCUT        = 22000  # Hz # High cut for our butter bandpass filter

    SILENCE_THRESH  = 50
    FRAME_RATE      = 44100

    IMG_W, IMG_H    = 344, 256

    MODEL           = 'model_26_8.h5'

    NB_SAMPLES      = 200
