# For the Dataset
TEXT_PATH="Dataset\\TR2EN_SEPERATED.txt"

# For data clip
START_RANGE=0
STOP_RANGE=3000

# FOR DATA PREPROCESSING
LANGUAGE_IN="turkish"
LANGUAGE_OUT="english"
START_SYMBOL="<SOS>"
STOP_SYMBOL="<EOS>"
PAD_SYMBOL="<PAD>"
MAX_LEN_SEQ=12

# FOR RANDOM SPLIT
VALID_RANGE=0.2

# For the Checkpoint
CALLBACKS_PATH="Transformer_checkpoint.pth.tar"
LOAD_CALLBACKS=False

# For The Training
TRAIN=True
EPOCHS=1
BATCH_SIZE=128
LEARNING_RATE=0.001
BETAS=(0.9,0.98)
EPSILON=(1e-09)

# For The Model
D_MODEL=512
NUM_HEADS=8
PAD_IDX=0
MASKING_VALUE=-1e8
NX=6

# FOR TEST
TEST=True

# FOR PREDICTION
PREDICTION=True
PREDICTION_SENTENCE="Merhaba"




