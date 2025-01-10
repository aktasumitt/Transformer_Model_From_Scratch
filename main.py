import Layers,Encoder,Decoder,Transformer,dataset,config,train,Checkpoints,Test,Prediction
import torch
from torch.utils.tensorboard import SummaryWriter


# Devices
devices=("cuda" if torch.cuda.is_available() else "cpu")


# Tensorboard
Tensorboard=SummaryWriter(config.TENSORBOARD_PATH,"Tensorboard Transformer Model")


# Loading Data and Max Length of All Sentences:
Turkish_sentences_list,English_sentences_list,MAX_LEN=dataset.Loading_Dataset(text_path=config.TEXT_PATH)


# Tokenizing Data and Adding Start, Stop, and Pad Symbol
turkish_tokenized=dataset.Preprocessing_Text(language="turkish",pad_symbol=config.PAD_TOKEN,max_len=MAX_LEN)(Turkish_sentences_list)
english_tokenized=dataset.Preprocessing_Text(language="english",start_symbol=config.START_TOKEN,stop_symbol=config.STOP_TOKEN,pad_symbol=config.PAD_TOKEN,max_len=MAX_LEN)(English_sentences_list) 


# Change tokens from word to idx
Tokenized_ENG,WORD2IDX_DİCT_ENG=dataset.WORD2IDX(start_sym=config.START_TOKEN,stop_sym=config.STOP_TOKEN,pad_sym=config.PAD_TOKEN)(english_tokenized)
Tokenized_TR,WORD2IDX_DİCT_TR=dataset.WORD2IDX(pad_sym=config.PAD_TOKEN)(turkish_tokenized)


# Calculate max number of tokens
NUM_TOKEN=max(len(WORD2IDX_DİCT_ENG),len(WORD2IDX_DİCT_TR))


# Create Dataset
dataset_total=dataset.Dataset(input_data=Tokenized_TR,word2idx_in=WORD2IDX_DİCT_TR,output_data=Tokenized_ENG,word2idx_out=WORD2IDX_DİCT_ENG)


# Random Split
Train_Dataset,Valid_Dataset,Test_Dataset=dataset.RandomSplit(dataset_total,valid_rate=config.VALID_RATE,test_rate=config.TEST_RATE)


# Dataloader
Train_Dataloader,Valid_Dataloader,Test_Dataloader=dataset.Dataloader(Train_Dataset,Valid_Dataset,Test_Dataset,batch_size=config.BATCH_SIZE)


# Model
ModelTransformer=Transformer.Transformer(Encoder=Encoder.Encoder,Decoder=Decoder.Decoder,Embedding=Layers.Embedding,
                                         PositionalEncoding=Layers.PositionalEncoding,MultiHeadAttention=Layers.MultiHeadAttention,
                                         FeedForward=Layers.FeedForward,d_model=config.D_MODEL,dk_model=config.DK_MODEL,batch_size=config.BATCH_SIZE,
                                         max_len=MAX_LEN,num_token=NUM_TOKEN,Nx=config.NX,devices=devices,
                                         STOP_TOKEN=WORD2IDX_DİCT_ENG[config.STOP_TOKEN]).to(devices)


# Optimizer and Loss
optimizer=torch.optim.Adam(params=ModelTransformer.parameters(),lr=config.LEARNING_RATE,betas=config.BETAS,eps=config.EPSILON)
loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1)


# Loading checkpoint if you have (if LOAD==True)
INITIAL_EPOCH=Checkpoints.Load_Checkpoint(LOAD=config.LOAD_CHECKPOINT,checkpoint_dir=config.CHECKPOINT_PATH,model=ModelTransformer,optimizer=optimizer)


# Train
if config.TRAIN==True:
    train.Train(Train_dataloader=Train_Dataloader,Valid_dataloader=Valid_Dataloader,EPOCH=config.EPOCH,Initial_epoch=INITIAL_EPOCH,
                Model=ModelTransformer,optimizer=optimizer,loss_fn=loss_fn,devices=devices,save_checkpoint_fn=Checkpoints.Save_Checkpoint,
                checkpoint_path=config.CHECKPOINT_PATH,Tensorboard=Tensorboard)

# Test
if config.TEST==True:
    Test.Test(Test_Dataloader=Test_Dataloader,devices=devices,Model=ModelTransformer,loss_fn=loss_fn)


# Prediciton
if config.PREDICTION==True:
    translate=Prediction.Prediction(Sentence=config.PREDICTION_SENTENCE,max_len=MAX_LEN,PAD_TOKEN=config.PAD_TOKEN,START_TOKEN=config.START_TOKEN,
                                    word2idx_dict_tr=WORD2IDX_DİCT_TR,batch_size=config.BATCH_SIZE,devices=devices)