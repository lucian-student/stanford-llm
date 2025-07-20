from cs336_basics.tokenizer import Tokenizer
import os
import multiprocessing

def main():
    multiprocessing.set_start_method("spawn", force=True)
    path = os.path.join(os.path.dirname(__file__),"..","tests","fixtures","corpus.en")
    tokenizer = Tokenizer(vocab_size=500,special_tokens=["<|endoftext|>"])   
    tokenizer.fit(path,verbose=True,num_processes=5)
    print(tokenizer.merges)

if __name__ == "__main__":
    main()