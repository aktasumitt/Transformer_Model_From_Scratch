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
        self.max_len_decoder=max_seq_len_decoder
        self.batch_size=batch_size
        self.max_len_decoder=max_seq_len_decoder
        self.stop_token=stop_token

        self.encoder=Encoder_Model(d_model=d_model,vocab_size=vocab_size_encoder,num_heads=num_heads,pad_idx=pad_idx,
                                   max_seq_len=max_seq_len_encoder,devices=devices,N_repeat=Nx)
        
        self.decoder=Decoder_Model(d_model=d_model,vocab_size=vocab_size_decoder,num_heads=num_heads,pad_idx=pad_idx,
                                   max_seq_len=max_seq_len_decoder,devices=devices,batch_size=batch_size,masking_value=masking_value
                                   ,N_repeat=Nx)
        
        
        self.linear=nn.Linear(d_model*max_seq_len_decoder,vocab_size_decoder)
        
        
    def forward(self,encoder_input,decoder_target=None):
        stop_step=0
        
        # input decoder
        input_decoder=torch.zeros((self.batch_size,self.max_len_decoder),dtype=torch.int).to(self.devices)
        input_decoder[:,0]=1 # Add Start token <SOS> to input decoder sentence = [1,0,0,0,0...]
        
        output_list=[]
        
        # Train Encoder Model
        encoder_out=self.encoder(encoder_input)
        
        for step in range(0,self.max_len_decoder-1):
            
                input_decoder_clone=input_decoder.clone()
                
                # Train Decoder Model
                decoder_out=self.decoder((encoder_out,input_decoder_clone))
                out=self.linear(decoder_out)
                
                # Output list        
                output_list.append(out)
                

                # FOR THE PREDICTION
                if decoder_target==None:
                    
                    # Prediction words for adding input_encoder
                    out_soft=torch.nn.functional.softmax(out,dim=-1)
                    _,pred=torch.max(out_soft,-1)
                    
                    # add pred words to input_decoder .       
                    input_decoder[:,step+1] = pred
                
                
                # FOR THE TRAINING
                else:
                    input_decoder[:,step+1]= decoder_target[:,step]
                    
                    # if after word is stop token, it will break.
                    stop_step+=(decoder_target[:,step]==self.stop_token).sum().item()
                    
                    if stop_step==self.batch_size:
                        break
                    
                
        # return output list and step for calculating loss and backward
        return torch.stack(output_list).permute(1,0,2),step
        
        
