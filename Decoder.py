import torch.nn as nn
import torch



class Decoder_Model(nn.Module):
    """
    Decoder_Model input to forward is tuple that have encoder_output and decoder_input like that (encoder_output,decoder_input)
    """
   
    def __init__(self,devices,d_model:int,vocab_size:int,num_heads:int,pad_idx:int,max_seq_len:int,batch_size:int,masking_value:int=-1e8,N_repeat:int=6):
        super(Decoder_Model,self).__init__()
        
        self.batch_size=batch_size          # Batch_size of model
        self.masking_value=masking_value    # masking value that is too big value on inf
        self.devices=devices                # device "cuda or cpu"
        self.d_model=d_model                # d_model for model layers total unit size => (512 in article )
        self.num_heads=num_heads                # num_heads is number of head (repetitions) for query, value and key
        self.max_seq_len=max_seq_len            # Maximum lenghts of sentences
        self.d_k=int(self.d_model/num_heads)    # d_k is for scaled dot product and (query, key, value) unit size
        self.Nx=N_repeat                        # Nx is the number of repetitions for decoder => (6 in article)
        
        # Embedding
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model,padding_idx=pad_idx)
        
        # Masked Multi Head Attention
        self.query_m=torch.nn.Linear(in_features=self.d_model,out_features=self.d_k) 
        self.key_m=torch.nn.Linear(in_features=self.d_model,out_features=self.d_k)     
        self.value_m=torch.nn.Linear(in_features=self.d_model,out_features=self.d_k)
        self.concat_scaled_dot_product_m=torch.nn.Linear(512,512) 

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
        PE_full=PE_full.to(self.devices)

        
        return PE_full.to(self.devices)
    
   
    # Masked Multi Head Attention
    def Masked_Multihead_Attention(self,data):
    
        # We Use masked with masking_value that is too big negative or positive
        mask=torch.triu(torch.ones(self.batch_size,self.max_seq_len,self.max_seq_len),diagonal=1).to(self.devices)
        mask_data=self.masking_value*mask
        
        scaled_dot_product=[]
        
        for i in range(self.num_heads):
            
            query=self.query_m(data)    #(batch,10,64) 10=max_len
            key=self.key_m(data)      
            value=self.value_m(data)
            
            dot_product=torch.matmul(query,torch.transpose(key,1,2))/(self.d_k**1/2) #(batch,10,10) 
            
            # Masked
            masked_product=mask_data+dot_product # (batch,10,10)
            
            scaled_dot=torch.nn.functional.softmax(masked_product,dim=2)  #(batch,10,10)
            scaled_dot=torch.matmul(scaled_dot,value)    #(batch,10,64)

            scaled_dot_product.append(scaled_dot)
        
        concat_scaled_dot_product=torch.concat(scaled_dot_product,dim=2) #(batch,10,512)
        concat_scaled_dot_product=self.concat_scaled_dot_product_m(concat_scaled_dot_product) #(batch,10,512)

        return concat_scaled_dot_product
    
    
    # Multi Head Attention
    def Multihead_Attention(self,data,encoder_out):

        scaled_dot_product=[]
        
        for i in range(self.num_heads):
            
            query=self.query(data)    
            key=self.key(encoder_out)      
            value=self.value(encoder_out)
            
            dot_product=torch.matmul(query,torch.transpose(key,1,2))/((self.d_model/self.num_heads)**1/2) 
            scaled_dot=torch.nn.functional.softmax(dot_product,dim=2)              
            scaled_dot=torch.matmul(scaled_dot,value)    

            scaled_dot_product.append(scaled_dot)
        
        concat_scaled_dot_product=torch.concat(scaled_dot_product,dim=2)
        concat_scaled_dot_product=self.concat_scaled_dot_product(concat_scaled_dot_product) 
        
        return concat_scaled_dot_product
    
    
    # Feed Forward
    def Feed_Forward(self,data):
    
        data=self.feed_forward1(data)
        data=torch.nn.functional.relu(data)
        data=self.feed_forward2(data)
        
        return data
    
    
    def forward(self,data):
        encoder_out,data_dec=data
        
        embed=self.embedding(data_dec)
        positional_enc=self.Positional_Encoding()
        data=embed+positional_enc
        
        for i in range(self.Nx) :
            
            mmhe_data=self.Masked_Multihead_Attention(data=data)
            norm_mmhe=nn.functional.layer_norm((mmhe_data+data),normalized_shape=mmhe_data.shape)

            mhe_data=self.Multihead_Attention(data=norm_mmhe,encoder_out=encoder_out)
            norm_mhe=nn.functional.layer_norm((mhe_data+data),normalized_shape=mhe_data.shape)
            
            feed_forward=self.Feed_Forward(data=norm_mhe)
            data=nn.functional.layer_norm((norm_mhe+feed_forward),normalized_shape=feed_forward.shape)
        
        return data.view(-1,self.max_seq_len*self.d_model)


        