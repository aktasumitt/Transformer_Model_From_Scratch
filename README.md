# Transformer Model From Scratch for Translate 

## Introduction:
In this project, I aimed to train a Custom Transformer model for translate text from Turkish to English.

## Tensorboard:
TensorBoard, along with saving training or prediction images, allows you to save them in TensorBoard and examine the changes graphically during the training phase by recording scalar values such as loss and accuracy. It's a very useful and practical tool.

## Dataset:
- In this project, I used a dataset containing 430,000 sentences with corresponding English and Turkish translations.
-  However, I haven't trained it yet because it requires a very high amount of time, so I cannot share my training data here,
-  But everything will be ready for those who want to train it. When tested, it works flawlessly.

## Model:
- Although originally designed for translation, the Transformer serves as the foundation for many current generator models and can produce results very close to human-like. Therefore, it represents a revolutionary advancement.

## Embedding:
- Before input tokenized sentences are fed into the both encoder and decoder model, they are passed through an embedding layer, to which positional encoding values are added.
- - Since the order of words in a sentence is important, the positional encoding layer maintains these sequences using sine and cosine functions.

#### Encoder:
- The Transformer encoder and decoder are advanced sequence-to-sequence generator models.
- Subsequently, Embedding output is passed to the encoder layer, which consists of a multi-head attention and a feed-forward layer in sequence, with layer normalization and residual connections in between.
- Multi-head attention is the application of self-attention at multiple steps, and the output is concatenated.
- Through these operations, we effectively capture the connections between words. 
- The output of the encoder is used in computing the query and value for the decoder's multi-head attention.

#### Decoder:
- The decoder consists of masked multi-head attention, multi-head attention, and feed-forward layers in sequence, with layer normalization and residual connections in between.
-  The decoder follows an autoregressive structure, meaning the outputs are calculated by adding the input values at each step. 
-  In the masked step, a large or infinite value is selected to mask the data during self-attention. This is the difference from multi-head attention.
-  The output generated in the decoder is passed through a linear layer and then softmax to predict the next word.
-  This process continues until a stop token is appended to the end of the sentence.
-  Instead, teacher forcing can be used, where the model adds the correct tokens to the input at each step instead of its own outputs. This allows us to reach the correct result faster
- After reaching the stop token, the sentence is taken, and the loss is calculated with the last decoder output for the training process to continue in a classic manner. 
- The Encoder and Decoder iterate multiple times consecutively, typically 6 times as mentioned in the paper.

#### For More details:
- Model: https://arxiv.org/abs/1706.03762
- Dataset : https://www.kaggle.com/datasets/seymasa/turkish-to-english-translation-dataset

## Train:
- With the resources at my disposal, I haven't been able to train the model further because it requires a lot of time and resources. 
- However, when tested on a small dataset with low epochs, the model was working smoothly. 
- For training, I used Turkish sentence data as input and English translations as output.
- I employed the Adam optimizer and CrossEntropyLoss for loss calculation. 
- The Turkish tokenized data is first fed into the encoder to produce output, then padded tensors with ("1") as the start token and the rest padded with ("0") are passed to the decoder.
- The generated last output are then summed up, and the loss is calculated in the loss function. 
- Since we used teacher forcing, instead of appending the outputs to the input data, we continued training by appending the actual expected outputs, which accelerates convergence to the correct result

## Prediction:
- The given text is first converted to tokens using a word-to-idx dictionary, then fed into the model.
- Afterwards, the resulting output is converted back to words using idx-to-word mapping. This completes the prediction process.

## Usage: 
- You can train the model by setting "TRAIN" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Tensorboard files will created into "runs" folder during training time.
- Then you can generate the translate sentences by setting the "LOAD_CALLBACKS" and "TEST" values to "True" in the config file.

