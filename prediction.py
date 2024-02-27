import torch,nltk

def prediction(sentence,Model,word2idx_in,padding_len,pad_idx,devices,batch_size):
    
    inputs=[]
    input_padded=[]
    
    # Tokenize with nltk
    for words in nltk.tokenize.word_tokenize(sentence.lower(),language="turkish"):
        inputs.append(word2idx_in[words])
    
    # Padding according to Model
    input_padded.append(list(nltk.pad_sequence(sequence=inputs,n=(padding_len-len(inputs)+1),pad_right=True,pad_left=False,right_pad_symbol=pad_idx)))

    # To give model
    with torch.no_grad():
        out,step=Model(torch.tensor(input_padded).repeat(batch_size,1).to(devices))
    
    # Prediction
    _,pred=torch.max(out,-1)
        
    pred_list=[]
    
    # IDX TO WORD 
    idx2word_out={v:k for k,v in word2idx_in.items()}
    
    for token_pred in pred[0]:
        pred_list.append(idx2word_out[token_pred.item()])
    
    
    print("Input_sentences: ",sentence)
    print("Translate: ", " ".join(pred_list))
    
    return pred_list
    
    