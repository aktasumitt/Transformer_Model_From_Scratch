import nltk,torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.transforms import VocabTransform
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset


# Loading Dataa
def Loading_Dataset(Text_Path, start_range: int = 0, stop_range: int = -1):

    #   This tokenizer need to your text dataset's sentences of languages seperate with tab (\\t)
    #   For Example: (Ben bir öğrenciyim. \\t I am a student.)

    print("Dataset is loading...")
    with open(Text_Path, "r", encoding="utf-8") as dosya:
        Text = dosya.readlines()

    Text = Text[start_range:stop_range]

    Sentences1 = []
    Sentences2 = []

    for i in Text:
        Sentences1.append(i.split("\t")[1][:-1])
        Sentences2.append(i.split("\t")[0])

    return Sentences1, Sentences2




# Preproccess text datas
class Preproccess():

    """
    We can tokenize and pad text datas with this class. 
    """

    def __init__(self, max_len: int, language: str, start_symbol: str = None, stop_symbol: str = None, pad_symbol: str = None):

        self.start_symbol = start_symbol
        self.stop_symbol = stop_symbol
        self.pad_symbol = pad_symbol
        self.max_len = max_len
        self.language = language

    # Tokenize Texts
    def Tokenize(self, Text):

        Tokenized = []

        for i in Text:
            timestep_list = nltk.tokenize.word_tokenize(
                i.lower(), language=self.language)

            if self.start_symbol != None:
                timestep_list.insert(0, self.start_symbol)
            if self.stop_symbol != None:
                timestep_list.append(self.stop_symbol)

            Tokenized.append(timestep_list)

        return Tokenized

    
    # Padding if u want
    def Padding_Text(self, Tokenized_data):

        padded_list = []
        for i in Tokenized_data:
            padded_list.append(list(nltk.pad_sequence(sequence=i, n=(
                self.max_len-len(i)+1), pad_right=True, pad_left=False, right_pad_symbol=self.pad_symbol)))
        return padded_list

    
    # call class
    def __call__(self, text):
        
        print(f"{self.language} Texts are Preprocessing...")

        data_prep = self.Tokenize(Text=text)

        if self.pad_symbol != None:
            data_prep = self.Padding_Text(data_prep)

        return data_prep




class Create_Dataset(Dataset):
    def __init__(self, tokenized_data_in, tokenized_data_out):
        super().__init__()
        self.tokenized_data_in = tokenized_data_in
        self.tokenized_data_out = tokenized_data_out

        self.vocab_in = build_vocab_from_iterator(
            self.tokenized_data_in, special_first=True, specials=["<PAD>", "<SOS>", "<EOS>", "<UNK>"])
        self.vocab_out = build_vocab_from_iterator(
            self.tokenized_data_out, special_first=True, specials=["<PAD>", "<SOS>", "<EOS>", "<UNK>"])

        self.data_in = VocabTransform(self.vocab_in)(self.tokenized_data_in)
        self.data_out = VocabTransform(self.vocab_out)(self.tokenized_data_out)

    def __len__(self):
        return len(self.tokenized_data_in)

    def word2idx_out(self):
        # creating word2idx dict for output
        return self.vocab_out.get_stoi()

    def word2idx_input(self):
        # creating word2idx dict for input
        return self.vocab_in.get_stoi()

    def __getitem__(self, index):

        return (torch.tensor(self.data_in[index]), torch.tensor(self.data_out[index]))





# Random split
def random_split_fn(dataset, valid_range):
    valid_size = int(len(dataset)*valid_range)

    Train, Valid, Test = random_split(
        dataset, [len(dataset)-(valid_size*2), valid_size, valid_size])
    return Train, Valid, Test




# Dataloader
def Dataloader_fn(train, valid, test, batch_size):

    train = DataLoader(dataset=train, batch_size=batch_size,
                       shuffle=True, drop_last=True)
    valid = DataLoader(dataset=valid, batch_size=batch_size,
                       shuffle=False, drop_last=True)
    test = DataLoader(dataset=test, batch_size=batch_size,
                      shuffle=False, drop_last=True)

    return train, valid, test
