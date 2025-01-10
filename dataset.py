import torch
import nltk
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform
from torch.utils.data import Dataset,DataLoader,random_split

# Loading Dataset
def Loading_Dataset(text_path:str,start_range:int=0,stop_range:int=10000):
    
    
    with open(text_path,"r",encoding="utf-8") as dosya:
        
        Text=dosya.readlines()
    
    English_Sentences_List=[]
    Turkish_sentence_list=[]
    MAX_LEN=0
    
    print("Dataset is loading...\n")
    
    for step,sentences in enumerate(Text):
        
        if step>=start_range and step<=stop_range:
            
            english_sentence,turkish_sentence=sentences.split("\t")
            
            # Calculate Max Len and Append list the sentences
            MAX_LEN=max(len(english_sentence),len(turkish_sentence),MAX_LEN) # calculate maximum length of sentences
                
            English_Sentences_List.append(english_sentence)
            Turkish_sentence_list.append(turkish_sentence[:-1])
    
    return Turkish_sentence_list,English_Sentences_List,MAX_LEN



# Tokenize sentences. 
# Add; pad, start and stop symbol 
class Preprocessing_Text():
    def __init__(self,language:str,max_len:int,start_symbol:str=None,stop_symbol:str=None,pad_symbol:str=None):
        
        self.language=language
        self.max_len=max_len
        self.start_sym=start_symbol
        self.stop_sym=stop_symbol
        self.pad_sym=pad_symbol
        
    def Tokenize(self,text_list):
        
        Tokenized_list=[]
        
        for sentence in text_list:
            
            # Tokenize sentences
            tokenize=nltk.tokenize.word_tokenize(sentence.lower(),language=self.language)
            
            # Add start symbol in first index
            if self.start_sym!=None:
                tokenize.insert(0,self.start_sym)
            
            # Add stop symbol in last index
            if self.stop_sym!=None:
                tokenize.append(self.stop_sym)
                
            Tokenized_list.append(tokenize)
        
        return Tokenized_list

    
    def Padding(self,tokenized_list):
        
        Padding_list=[]
        
        # Add pad symbol till max length
        for sentence in tokenized_list:
            
            pad_len=self.max_len-len(sentence)+1
            padded=nltk.pad_sequence(sentence,n=pad_len,pad_right=True,right_pad_symbol=self.pad_sym)
            
            Padding_list.append(list(padded))
        
        return Padding_list
    
    
    def __call__(self,text):
        tokenized=self.Tokenize(text_list=text)
        padded=self.Padding(tokenized)
    
        return padded



# Tokenized word to idx
class WORD2IDX():
    
    def __init__(self,start_sym:str=None,stop_sym:str=None,pad_sym:str=None,unknown_sym:str=None):
        
        self.start_sym=start_sym
        self.stop_sym=stop_sym
        self.pad_sym=pad_sym
        
    def Word2idx(self,text):
        
        if self.start_sym==None and self.stop_sym==None:
            special_tokens=[self.pad_sym]
        else:
            special_tokens=[self.pad_sym,self.start_sym,self.stop_sym]
        
        builded=build_vocab_from_iterator(text,special_first=True,specials=special_tokens)
        Tokenized_text=VocabTransform(builded)(text)
        WORD2IDX_dict=builded.get_stoi()
        
        return Tokenized_text,WORD2IDX_dict
    
    def __call__(self,text):
        
        Tokenized,WORD2IDX_DİCT=self.Word2idx(text)
        return Tokenized,WORD2IDX_DİCT
        


# Dataset Creating
class Dataset(Dataset):
    def __init__(self,input_data,word2idx_in,output_data:None,word2idx_out:None):
        super(Dataset,self).__init__()
        self.input_data=torch.tensor(input_data)
        self.output_data=torch.tensor(output_data)
        self.idx2word_in={v:k for (k, v) in word2idx_in.items()}
        self.idx2word_out={v:k for (k, v) in word2idx_out.items()}
    
    def __len__(self):
        return len(self.input_data)
        
    def __getitem__(self, idx):
        
        if self.output_data==None: # For Prediction we wont have output data
            return self.input_data[idx]
        else:
            return (self.input_data[idx],self.output_data[idx])
     
     
# Ranadom split
def RandomSplit(data,valid_rate,test_rate):
    
    valid_size=int(valid_rate*len(data))
    test_size=int(test_rate*len(data))
    train_size=len(data)-(valid_size+test_size)
    
    Train,Valid,Test=random_split(data,[train_size,valid_size,test_size])
    
    return Train,Valid,Test

# Dataloader
def Dataloader(train,valid,test,batch_size):
    
    Train=DataLoader(train,batch_size,shuffle=True,drop_last=True)
    Valid=DataLoader(valid,batch_size,shuffle=False,drop_last=True)
    Test=DataLoader(test,batch_size,shuffle=False,drop_last=True)
    
    return Train,Valid,Test


        
        
