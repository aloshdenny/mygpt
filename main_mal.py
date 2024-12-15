import asyncio
import json
import torch
import sentencepiece as spm
from model import GPTModel, get_batch, device
from contextlib import asynccontextmanager

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file='malayalam_tokenizer.model')

m = None

@asynccontextmanager
async def lifespan():
    global sp, m
    # Load the ML model
    model = GPTModel(sp.vocab_size())
    m = model.to(device)
    print("Loading model...")
    m.load_state_dict(torch.load("mine.pt"))

    yield
    # Clean up the ML models and release the resources

class CreateCompletionRequest:
    def __init__(self, prompt, stream=False, stop=None, max_tokens=100, temperature=1.0):
        self.prompt = prompt
        self.stream = stream
        self.stop = stop or []
        self.max_tokens = max_tokens
        self.temperature = temperature

def on_generated(token):
    print(sp.decode([token.item()]), end="")

def completions(body: CreateCompletionRequest):
    """ Mimic the Llama-CPP server's completion functionality """
    global sp, m

    # Encode stops using SentencePiece
    stops = [{"choices": [{"text": sp.encode(stop)}]} for stop in body.stop]

    # Encode the prompt
    prompt_tokens = sp.encode(body.prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    ret = m.generate(idx, max_new_tokens=body.max_tokens, stop_tokens=stops, generated=on_generated)

    tokens = ret.tolist()[0]

    if body.stream:
        # This is a terrible hack to do response streaming: just wait until it is done and then dribble out the tokens.
        chunks = [json.dumps({"choices": [{"text": sp.decode([token])}]}) for token in tokens]

        async def event_generator():
            while True:
                if len(chunks) == 0:
                    break
                yield chunks.pop(0)
                await asyncio.sleep(0.01)

        return event_generator()  # Return generator function for streaming response
    else:
        return {"resp": sp.decode(tokens)}

def text_from_path(path):
    """ Load all the text files found at this path into one huge lump of text and return it. """
    import glob
    import os

    text = ""
    print(f"Searching for text files in: {os.path.abspath(path)}")
    files = glob.glob(path+"*.txt")

    print(f"Found {len(files)} text files:")
    for filename in files:
        print(filename)
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    text += line
        except UnicodeDecodeError:
            print(f"Warning: Could not read {filename} with UTF-8 encoding. Trying alternative encodings.")
            try:
                with open(filename, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                    for line in lines:
                        text += line
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print(f"Total text length: {len(text)} characters")
    return text

def text_from_file():
    """ Open a single file and return its contents. """
    text = ""
    with open('input.txt') as f:
        lines = f.readlines()
        for line in lines:
            text += line
    return text

def train():
    """ Train the model and save it at regular checkpoints. """
    global sp, m

    # Change this path to where your training data is, as a folder full of .txt files
    with open('input_subset.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Split the text into lines and limit to the first 10,000 lines
    lines = text.splitlines()
    limited_lines = lines[:10000]  # Only take the first 10,000 lines
    limited_text = "\n".join(limited_lines)

    tokens = torch.tensor(sp.encode(text), dtype=torch.long)

    print(f"Vocab size {sp.vocab_size()}")
    print(f"Training data size {len(tokens)}")

    model = GPTModel(sp.vocab_size())
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # Try to load the old model so we can continue training it from where we left off.
    try:
        m.load_state_dict(torch.load("mine.pt"))
    except FileNotFoundError:
        print("Model file not found, starting new training...")
    print("Loaded model. Beginning training...")

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    epoch = 300
    for checkpoints in range(10):
        loss_total = 0
        for steps in range(epoch):
            xb, yb = get_batch(tokens)
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(loss_total/epoch)

        torch.save(model.state_dict(), "mine.pt")
    print("Done with training!")

def inference():
    """ Generate a stream of text from a starting prompt. """
    global m, sp

    model = GPTModel(sp.vocab_size())
    m = model.to(device)

    # This is the model that was trained previously. The hyperparameters in model.py must match exactly to when it was trained, or there'll be an error.
    print("Loading model...")
    m.load_state_dict(torch.load("mine.pt"))

    def on_generated(token):
        # Ensure that the token is passed correctly to sp.decode() as a list
        print(sp.decode([token.item()]), end="")

    stops = [sp.encode(".")[0], sp.encode("?")[0], sp.encode("!")[0]]
    prompt = "പുറത്ത്‌ മഞ്ഞുമഴ"
    print(f"{prompt}", end="")
    prompt_tokens = sp.encode(prompt)
    # supress = torch.tensor([0, 930], dtype=torch.long, device=device) # the and and: 290, 262 as a testy=
    supress = []
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    m.generate(idx, max_new_tokens=100, stop_tokens=None, generated=on_generated, top_k=16, sample=True, supress_tokens=supress)
    print("\n")


if __name__ == '__main__':
    print(f"Using device {device}")

    # Uncomment this to train the model
    train()

    # Uncomment this for inference testing
    inference()