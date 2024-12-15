# mygpt
An easily-trained baby GPT that can stand in for the real thing. Based on Andrej Karpathy's makemore, but set up to mimic a llama-cpp server.

The main points of differentiation are:
 - my version is token-based (I use tiktoken for english and sentencepiece for malayalam)
 - code to load up multiple text files as a training set
 - extra inference parameters, such as top_k, and the supression of tokens which you do not want to see (ie glitch tokens or annoyingly repeated tokens).

So you can train the default tiny 15M parameter model, and use that in your projects instead of ChatGPT.

This is not production-ready; it's a toy implementation for educational purposes.

## Setup

pip install -r requirements.txt

## Using it

Run main.py for training a 15M English GPT, although it's not very configurable. What you're looking for is at the bottom of this readme.

### Training

Uncomment "train()" in main.py / main_mal.py. It will save checkpoints of the model parameters into the CWD.

### Inference / text generation

Once you have trained the model, comment "train()" and uncomment "inference()". Setup whatever prompt you want. Then run the script to see the generated text appear.

[Now try this for Malayalam!](https://colab.research.google.com/drive/1kWZ9CqXXcZpAYvaxQ7SqrgaETTk0Y5vX?usp=sharing)
