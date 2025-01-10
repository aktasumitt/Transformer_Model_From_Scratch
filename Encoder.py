import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,MultiHeadAttention,FeedForward,d_model,dk_model,batch_size,max_len):
        super(Encoder,self).__init__()
        
        self.layer_norm=nn.LayerNorm(d_model)
        self.MHA_encoder=MultiHeadAttention(d_model,batch_size,dk_model,max_len,MASK=False)
        self.feed_forward_encoder=FeedForward(d_model)
                
        
    def forward(self,encoder_in):
        
        mhe_out=self.MHA_encoder(query_data=encoder_in,key_data=encoder_in,value_data=encoder_in) # MHA
        mhe_out=self.layer_norm(mhe_out+encoder_in) # add and layer_norm
        
        ff_out=self.feed_forward_encoder(mhe_out) # Feed Forward
        ff_out=self.layer_norm(ff_out+mhe_out) # add and layer_norm
        
        return ff_out