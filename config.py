# For the dataset
START_TOKEN="<SOS>"
STOP_TOKEN="<EOS>"
PAD_TOKEN="<PAD>"
VALID_RATE=0.25
TEST_RATE=0.15
BATCH_SIZE=100
NUM_TOKEN=1000

# For the Model
D_MODEL=512
DK_MODEL=64
NX=6

# For the Training
TRAIN=True
EPOCH=2
LEARNING_RATE=0.001
BETAS=(0.9,0.98)
EPSILON=(1e-09)

# Paths
TENSORBOARD_PATH="Tensorboard"
CHECKPOINT_PATH="checkpoints.pth.tar"
TEXT_PATH="Dataset\\TR2EN.txt"

# Load checkpoints if u have
LOAD_CHECKPOINT=False

# For the test
TEST=True

# For the Prediction
PREDICTION=True
PREDICTION_SENTENCE="Merhaba"




