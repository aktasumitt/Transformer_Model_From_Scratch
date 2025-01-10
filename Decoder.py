import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,MultiHeadAttention,FeedForward,
                 d_model,dk_model,batch_size,max_len):
        super(Decoder,self).__init__()
        
        self.layer_norm_dec=nn.LayerNorm(d_model) # Layer Norm
        self.MHA_decoder=MultiHeadAttention(d_model,batch_size,dk_model,max_len,MASK=False) # Not masked MHA
        self.Masked_MHA_decoder=MultiHeadAttention(d_model,batch_size,dk_model,max_len,MASK=True) # Masked MHA
        self.feed_forward_decoder=FeedForward(d_model) # Feed Forward
        
    def forward(self,decoder_in,encoder_out):
        
        mmhe_out=self.MHA_decoder(query_data=decoder_in,key_data=decoder_in,value_data=decoder_in) 
        mmhe_out=self.layer_norm_dec(mmhe_out+decoder_in) # add and layer_norm
        
        mhe_out=self.MHA_decoder(query_data=mmhe_out,key_data=encoder_out,value_data=encoder_out)
        mhe_out=self.layer_norm_dec(mmhe_out+mhe_out) # add and layer_norm
        
        ff_out=self.feed_forward_decoder(mhe_out)
        ff_out=self.layer_norm_dec(ff_out+mhe_out) # add and layer_norm
        
        return ff_out