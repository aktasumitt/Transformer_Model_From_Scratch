import torch
import torch.nn as nn


class Encoder_Model(nn.Module):
    
    def __init__(self,d_model,vocab_size,num_heads,pad_idx,max_seq_len,devices,N_repeat):
        super(Encoder_Model,self).__init__()
        
        self.devices=devices                # device "cuda or cpu"
        self.d_model=d_model                # d_model for model layers total unit size => (512 in article) 
        self.num_heads=num_heads            # num_heads is number of head (repetitions) for query, value and key
        self.max_seq_len=max_seq_len        # Maximum lenghts of sentences
        self.Nx=N_repeat                    # Nx is the number of repetitions for decoder => (6 in article)
        self.d_k=int(self.d_model/num_heads)    # d_k is for scaled dot product and (query, key, value) unit size
        
        # Embedding
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model,padding_idx=pad_idx)
        
        # Multihead Attention
        self.query=torch.nn.Linear(in_features=self.d_model,out_features=self.d_k) 
        self.key=torch.nn.Linear(in_features=self.d_model,out_features=self.d_k)     
        self.value=torch.nn.Linear(in_features=self.d_model,out_features=self.d_k)
        self.concat_scaled_dot_product=torch.nn.Linear(512,512) 

        # Feed Forward
        self.feed_forward1=torch.nn.Linear(self.d_model,2048)
        self.feed_forward2=torch.nn.Linear(2048,self.d_model)
        
    # Positional Encoding
    def Positional_Encoding(self):
        
        position=torch.arange(0,self.max_seq_len).reshape((self.max_seq_len,1))
        even_i=torch.arange(0,self.d_model,2) # çift sıra
        odd_i=torch.arange(1,self.d_model,2) # tek sıra
        
        pow_even=torch.pow(10000,(2*even_i/self.d_model))   
        pow_odd=torch.pow(10000,(2*odd_i/self.d_model)) 
        
        PE_even=torch.sin(position/pow_even)
        PE_odd=torch.cos(position/pow_odd)
        
        PE_full=torch.stack([PE_even,PE_odd],dim=2)
        PE_full=torch.flatten(PE_full,start_dim=1,end_dim=2)
        PE_full=torch.unsqueeze(PE_full,0)

        PE_full=PE_full.to(device=self.devices)
        
        return PE_full
    
    # Multihead Attention
    def Multihead_Attention(self,data):

        scaled_dot_product=[]
        
        for i in range(self.num_heads):
            
            query=self.query(data)    
            key=self.key(data)      
            value=self.value(data)
            
            dot_product=torch.matmul(query,torch.transpose(key,1,2))/((self.d_model/self.num_heads)**1/2) 
            scaled_dot=torch.nn.functional.softmax(dot_product,dim=2)              
            scaled_dot=torch.matmul(scaled_dot,value)    

            scaled_dot_product.append(scaled_dot)
        
        concat_scaled_dot_product=torch.concat(scaled_dot_product,dim=2)
        concat_scaled_dot_product=self.concat_scaled_dot_product(concat_scaled_dot_product) 
        
        return concat_scaled_dot_product
    
    # Feed FOrward
    def Feed_Forward(self,data):
    
        data=self.feed_forward1(data)
        data=torch.nn.functional.relu(data)
        data=self.feed_forward2(data)
        
        return data
    
    def forward(self,data):
                
        embed=self.embedding(data)
        positional_enc=self.Positional_Encoding()
        data=embed+positional_enc
        
        for i in range(self.Nx):
            
            mhe_data=self.Multihead_Attention(data=data)
            norm_data=nn.functional.layer_norm((mhe_data+data),normalized_shape=mhe_data.shape) # layer Norm and residual
            
            feed_forward=self.Feed_Forward(data=norm_data)
            data=nn.functional.layer_norm((norm_data+feed_forward),normalized_shape=feed_forward.shape) # layer norm and residual

            
        return data
        