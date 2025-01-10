import torch.nn as nn
import torch

class Transformer(nn.Module):
    def __init__(self, Encoder, Decoder, Embedding, PositionalEncoding, MultiHeadAttention, FeedForward, d_model, dk_model, batch_size, max_len, num_token, Nx, devices, STOP_TOKEN: int):
        super(Transformer, self).__init__()
        
        self.batch_size = batch_size
        self.Nx = Nx
        self.max_len = max_len
        self.stop_token = STOP_TOKEN
        self.d_model = d_model
        
        # Embedding layer for both encoder and decoder
        self.embedding = Embedding(d_model, token_size=num_token, pad_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len, devices)
        
        # Stack encoder and decoder NX times
        self.encoder = nn.ModuleList([Encoder(MultiHeadAttention, FeedForward, d_model, dk_model, batch_size, max_len) for _ in range(Nx)])
        self.decoder = nn.ModuleList([Decoder(MultiHeadAttention, FeedForward, d_model, dk_model, batch_size, max_len) for _ in range(Nx)])
        
        # Flatten and Output layer
        self.flatten = nn.Flatten(-2, -1)
        self.last_linear = nn.Linear(d_model * max_len, num_token)
    
    def Encoder_Stack(self, input):
        
        # ENCODER
        embed_enc = self.embedding(input)  # same embedding with decoder and scale with root(d_model)
        encoder_in = self.positional_encoding(embed_enc)
        
        for i in range(self.Nx):  # Stack Encoder
            encoder_in = self.encoder[i](encoder_in)
        
        return encoder_in
    
    def Decoder_Stack(self, input_decoder, encoder_out):
        
        # DECODER
        embed_dec = self.embedding(input_decoder)  # same embedding with encoder and scale with root(d_model)
        decoder_in = self.positional_encoding(embed_dec)
        
        for i in range(self.Nx):
            decoder_in = self.decoder[i](decoder_in, encoder_out)        
        
        return decoder_in
    
        
    def forward(self, input_encoder, input_decoder, targets_decoder=None):
        stop_token_idx = 0
        output_list = []
        
        # ENCODER
        encoder_out = self.Encoder_Stack(input_encoder)
        
        # DECODER
        for i in range(self.max_len - 1):
            decoder_out = self.Decoder_Stack(input_decoder, encoder_out)
            
            # Prediction layer
            flat_out = self.flatten(decoder_out)
            out_transformer = self.last_linear(flat_out)
            output_list.append(out_transformer)
            
            if targets_decoder is not None:  # For training and we can use Teacher Forcing method
                stop_token_idx += (targets_decoder[:, i] == self.stop_token).sum().item()
                if stop_token_idx == self.batch_size:
                    break
                else:
                    # Create updated tensor
                    input_decoder = input_decoder.clone()
                    input_decoder[:, i+1] = targets_decoder[:, i]
            else:
                soft = nn.functional.softmax(out_transformer, dim=-1)
                _, pred = torch.max(soft, -1)
                
                if pred.item() == self.stop_token:
                    break
                else: 
                    # Create updated tensor
                    input_decoder = input_decoder.clone()
                    input_decoder[:, i+1] = pred.item()
        
        return torch.stack(output_list).permute(1, 0, 2)  # (max_len, B, num_tokens) --> (B, max_len, num_tokens)
