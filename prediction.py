import torch,dataset

# Data process for input of prediction
def Preprocessing(Sentence:str,max_len,PAD_TOKEN,word2idx_dict_tr,batch_size):
        
        turkish_tokenized=dataset.Preprocessing_Text(language="turkish",pad_symbol=PAD_TOKEN,max_len=max_len)([Sentence])
        
        turkish_w2idx=[word2idx_dict_tr[i] for i in turkish_tokenized[0]]
        
        input_data=torch.tensor(turkish_w2idx).repeat(batch_size,1)
        return input_data



def Prediction(Sentence:str,Model,max_len,PAD_TOKEN,START_TOKEN,word2idx_dict_tr,batch_size,devices):
    
    with torch.no_grad():
        
        input_data=Preprocessing(Sentence=Sentence,max_len=max_len,PAD_TOKEN=PAD_TOKEN,word2idx_dict_tr=word2idx_dict_tr,baych_size=batch_size).to(devices)
        
        
        # To remove start token from out decoder
        initial_input_decoder = torch.zeros_like(input_data).to(devices)
        initial_input_decoder[:, 0] = word2idx_dict_tr[START_TOKEN]

        output_test = Model(input_data,initial_input_decoder)


        _, pred_test = torch.max(output_test, -1)
        
        idx2word_dict={v:k for k,v in word2idx_dict_tr.items()}
        pred_list=[]
        for i in pred_test[0]:
           pred_list.append(idx2word_dict[i.item()])
           
        translate_sentence = " ".join(pred_list)
        
        print(translate_sentence)
        
        return translate_sentence
            

