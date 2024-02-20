import dataset,Decoder,Encoder,Train,transformer,config,callbakcs,torch,test,prediction
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

# Tensorboard
Tensorboard_Writer=SummaryWriter()

# Devices to use cuda
devices=("cuda" if torch.cuda.is_available() else "cpu")

   
# Loading Datasets from scratch
Turkish_Text,English_Text=dataset.Loading_Dataset(Text_Path=config.TEXT_PATH,
                                                        start_range=config.START_RANGE,
                                                        stop_range=config.STOP_RANGE)

# Preprocess Texts:
Tokenized_Turkish=dataset.Preproccess(max_len=config.MAX_LEN_SEQ,
                                      language=config.LANGUAGE_IN,
                                      pad_symbol=config.PAD_SYMBOL)(text=Turkish_Text)
          

Tokenized_English=dataset.Preproccess(max_len=config.MAX_LEN_SEQ,
                                      language=config.LANGUAGE_OUT,
                                      stop_symbol=config.STOP_SYMBOL,
                                      pad_symbol=config.PAD_SYMBOL)(text=English_Text)

# Create Dataset
Train_dataset=dataset.Create_Dataset(tokenized_data_in=Tokenized_Turkish,
                                     tokenized_data_out=Tokenized_English)

# WORD to IDX DICTIONARIES
W2IDX_IN=Train_dataset.word2idx_input()
W2IDX_OUT=Train_dataset.word2idx_out()

    
# Random Split
Train_dataset,Valid_dataset,Test_dataset=dataset.random_split_fn(dataset=Train_dataset,
                                                                 valid_range=config.VALID_RANGE)

# Create Dataloaders:
Train_dataloader,Valid_Dataloader,Test_Dataloader=dataset.Dataloader_fn(train=Train_dataset,
                                                                        valid=Valid_dataset,
                                                                        test=Test_dataset,
                                                                        batch_size=config.BATCH_SIZE)
# Create Model:
Model=transformer.Transformer_Model(d_model=config.D_MODEL,
                                    Encoder_Model=Encoder.Encoder_Model,
                                    Decoder_Model=Decoder.Decoder_Model,
                                    vocab_size_encoder=len(W2IDX_IN),
                                    vocab_size_decoder=len(W2IDX_OUT),
                                    num_heads=config.NUM_HEADS,
                                    pad_idx=config.PAD_IDX,
                                    max_seq_len_decoder=config.MAX_LEN_SEQ,
                                    max_seq_len_encoder=config.MAX_LEN_SEQ,
                                    devices=devices,
                                    batch_size=config.BATCH_SIZE,
                                    masking_value=config.MASKING_VALUE,
                                    Nx=config.NX,
                                    stop_token=W2IDX_IN[config.STOP_SYMBOL])
Model.to(devices)
Model.train()


# Create Optimizer and Loss_fn
optimizers=torch.optim.Adam(params=Model.parameters(),lr=config.LEARNING_RATE,betas=config.BETAS,eps=config.EPSILON)
loss_fn=torch.nn.CrossEntropyLoss()


# Load Callbacks  
if config.LOAD_CALLBACKS==True:
    print("Callbacks are Loading...")
    checkpoint=torch.load(f=config.CALLBACKS_PATH)
    start_epoch=callbakcs.Load_Callbakcs(model=Model,optimizer=optimizers,checkpoint=checkpoint)  
else:
    start_epoch=0
        

# Training
if config.TRAIN==True:
    Train.train(Train_Dataloader=Train_dataloader,
                Valid_Dataloader=Valid_Dataloader,
                optimizer=optimizers,
                loss_fn=loss_fn,
                model=Model,
                epochs=config.EPOCHS,
                start_epochs=start_epoch,
                devices=devices,
                save_callbacks=callbakcs.Save_Callbacks,
                checkpoint_path=config.CALLBACKS_PATH,
                Tensorboard=Tensorboard_Writer)


# TEST
if config.TEST==True:
    test.Test(Model=Model,Test_dataloader=Test_Dataloader,loss_fn=loss_fn,devices=devices)
 

# PREDICTION   
if config.PREDICTION==True:
    translate=prediction.prediction(sentence=config.PREDICTION_SENTENCE,
                                    Model=Model,
                                    word2idx_in=W2IDX_IN,
                                    padding_len=config.MAX_LEN_SEQ,
                                    pad_idx=config.PAD_IDX,
                                    devices=devices)




