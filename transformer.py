import torch
import torch.nn as nn 
from Encoder  import Encoder_Model
from Decoder import Decoder_Model




class Transformer_Model(nn.Module):
    
    """
    Transformer model input is encoder input. Decoder input will created in forward.
    
    """
    
    def __init__(self,Encoder_Model,
                Decoder_Model,
                Embeddig_Model,
                d_model,
                vocab_size_encoder,
                vocab_size_decoder,
                num_heads,
                pad_idx,
                max_seq_len_encoder,
                max_seq_len_decoder,
                devices,
                batch_size,
                masking_value,
                Nx,
                stop_token):
            
        super(Transformer_Model,self).__init__()
        
        self.devices=devices
        self.max_len_encoder=max_seq_len_encoder
        self.max_len_decoder=max_seq_len_decoder
        self.batch_size=batch_size
        self.stop_token=stop_token
        self.vocab_size_decoder=vocab_size_decoder
        
        self.encoder_embedding=Embeddig_Model(vocab_size=vocab_size_encoder,d_model=d_model,pad_idx=pad_idx,
                                              devices=devices,max_seq_len=max_seq_len_encoder)
        self.decoder_embedding=Embeddig_Model(vocab_size=vocab_size_decoder,d_model=d_model,pad_idx=pad_idx,
                                              devices=devices,max_seq_len=max_seq_len_decoder)
        
        self.encoder=nn.ModuleList([Encoder_Model(d_model=d_model,num_heads=num_heads,devices=devices,batch_size=batch_size) for i in range(Nx)])
        self.decoder=nn.ModuleList([Decoder_Model(devices=devices,d_model=d_model,num_heads=num_heads,batch_size=batch_size,masking_value=masking_value) for i in range(Nx)])
        
        self.linear=nn.Linear(d_model,vocab_size_decoder)
        
        
    def forward(self,encoder_input,decoder_target=None):
        stop_step=0
        
        # input decoder is start token <SOS>
        input_decoder=torch.ones((self.batch_size,1),dtype=torch.int).to(self.devices)
        
        # encoder embedding
        encoder_embed=self.encoder_embedding(encoder_input)
        
        # Train Encoder Model with Module List
        for encoder in self.encoder:
            encoder_out=encoder(encoder_embed)
        
        for step in range(0,self.max_len_decoder-1):  # we train max_len times
                            
            # Train Decoder Embedding
            decoder_embed=self.decoder_embedding(input_decoder)
            
            for decoder in self.decoder:
                decoder_out=decoder((encoder_out,decoder_embed))
            
            out=self.linear(decoder_out)
                
            # FOR THE PREDICTION
            if decoder_target==None:
                    
                # Prediction last words for changing input_encoder
                out_soft=torch.nn.functional.softmax(out[:,step,:],dim=-1)
                _,pred=torch.max(out_soft,-1)
                    
                # change pred words to input_decoder .       
                input_decoder=torch.concat([input_decoder, pred.unsqueeze(1)],dim=-1)
                
                    
                # if after word is stop token, it will break.
                stop_step+=(pred==self.stop_token).sum().item()
                    
                if stop_step==self.batch_size:
                    break
                
                # FOR THE TRAINING
            else:
                input_decoder=torch.concat([input_decoder,decoder_target[:,step].unsqueeze(1)],dim=-1)
                    
                # if after word is stop token, it will break.
                stop_step+=(decoder_target[:,step]==self.stop_token).sum().item()
                    
                if stop_step==self.batch_size:
                    break
        
        return out,step
        
        
