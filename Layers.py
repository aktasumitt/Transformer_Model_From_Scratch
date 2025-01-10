import torch.nn as nn
import torch

# Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,batch_size,dk_model,max_len,MASK:bool):
        super(MultiHeadAttention,self).__init__()
        
        self.MASK=MASK # MASK variable for masked MHA
        self.max_len=max_len # Max lenght of sentences
        self.dk=dk_model # dk value for MHA
        self.batch_size=batch_size # Batch size
        self.d_model=d_model # d_model is hidden layer size for model
        
        # Query key and value layers
        self.q_layer=nn.Linear(d_model,d_model,bias=False)
        self.k_layer=nn.Linear(d_model,d_model,bias=False)
        self.v_layer=nn.Linear(d_model,d_model,bias=False)
        
        # Projection layer for output of MHA
        self.projection_layer=nn.Linear(d_model,d_model)
        
        self.dropout=nn.Dropout(0.1)
        
    
    def forward(self,query_data,key_data,value_data):
        
        # Query, key and value
        query=self.q_layer(query_data).reshape(self.batch_size,self.max_len,-1,self.dk).permute(0,2,1,3)
        key=self.k_layer(key_data).reshape(self.batch_size,self.max_len,-1,self.dk).permute(0,2,1,3)
        value=self.v_layer(value_data).reshape(self.batch_size,self.max_len,-1,self.dk).permute(0,2,1,3)
        
        # dot product      
        scaled_dot_product=torch.matmul(query,torch.transpose(key,dim0=-2,dim1=-1)) / (self.dk**(1/2)) # Q x (K).T / root(dk)
        
        # masking
        if self.MASK==True:
            mask_tensor=torch.triu(torch.ones_like(scaled_dot_product),diagonal=1)*(-(1e13)) # -1e13 is mask value that too low value like a -(infitive)
            scaled_dot_product=scaled_dot_product+mask_tensor
        
        # Attention_weights
        attention_weights=nn.functional.softmax(scaled_dot_product,-1)
        
        output_att=torch.matmul(attention_weights,value)
        
        # output     
        out_concat=output_att.permute(0,2,1,3).reshape_as(query_data) # Permute >> (B,8,max_len,64) ---> Concatinate >>>(B,max_len,8,64)--->(B,max_len,512)

        return self.dropout(self.projection_layer(out_concat))


# Feed forward (4x) 
class FeedForward(nn.Module):
    
    def __init__(self,d_model):
        super(FeedForward,self).__init__()
        
        self.FF1=nn.Linear(d_model,d_model*4)
        self.FF2=nn.Linear(d_model*4,d_model)
        self.dropout=nn.Dropout(0.1)
        
    def forward(self,data):
        
        ff1_out=nn.functional.relu((self.FF1(data)))
        
        return self.dropout(self.FF2(ff1_out))


# Embedding Layer
class Embedding(nn.Module):
    def __init__(self,d_model,token_size,pad_idx=0):
        super(Embedding,self).__init__()
        
        self.d_model=d_model
        self.embedding=nn.Embedding(num_embeddings=token_size,embedding_dim=d_model,padding_idx=pad_idx)
    
    def forward(self,data):
        return self.embedding(data)*(self.d_model**0.5) 


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len,devices):
        super(PositionalEncoding,self).__init__()
        
        self.devices=devices
        self.max_len=max_len
        self.d_model=d_model
        self.droput=nn.Dropout(0.1)
    
    def positional_encoding(self):
            
        # Positions of text sequences
        position=torch.arange(0,self.max_len,1).reshape(self.max_len,1).to(self.devices)

        # Even and odd tensors as embedding size
        even_i=torch.arange(0,self.d_model,2).to(self.devices)
        odd_i=torch.arange(0,self.d_model,2).to(self.devices)

        # Calculate power to use in sinus and cos fuction 
        even_pow=torch.pow(10000,(2*even_i)/self.d_model)
        odd_pow=torch.pow(10000,(2*odd_i)/self.d_model)
        
        # Sin and cos function to calculate position even and odd row
        PE_sin=torch.sin(position/even_pow)
        PE_cos=torch.cos(position/odd_pow)
        
        # Concat odd and even positions for reached positional encoding tensor
        positional_enc=torch.stack([PE_sin,PE_cos],dim=-1).flatten(start_dim=-2,end_dim=-1)
        return positional_enc
        
    def forward(self,embed):
        return self.droput(self.positional_encoding()+embed)
        
        
        
        
        
        
        
        
        
        


