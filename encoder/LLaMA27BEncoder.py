import llama
import os
from pathlib import Path
import pickle

class LLaMA27BEncoder:
    def __init__(self, model_name="llama-2-7b", tokenizer_path="tokenizer.model"):
        """
        Initializes the LLaMA-2-7B encoder with the specified model.
        """
        llama_path = Path(os.path.dirname(llama.__file__)).parent
        model_path = os.path.join(llama_path, model_name)
        tokenizer_path = os.path.join(llama_path, tokenizer_path)
        self.generator = llama.Llama.build(
            ckpt_dir=model_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=512,
            max_batch_size=6,
        )

    def encode(self, text, layer_nums=[15, 20, 25]):
        """
        Encodes the given text using the LLaMA-2-7B model.

        Args:
        text (str): The text to be encoded.

        Returns:
        list: The encoded text.
        """
        
        encoded_text = self.generator.encode_layer_output(text, layer_nums=[15, 20, 25])
        for i in range(len(encoded_text)):
            encoded_text[i] = encoded_text[i].cpu().detach().squeeze().numpy()
        return encoded_text
    
if __name__ == '__main__':
    '''
    usage: torchrun --nproc_per_node 1 -m encoder.LLaMA27BEncoder
    '''
    # Example usage
    llama_encoder = LLaMA27BEncoder()

    # Encode some text
    sample_text = "This is an object."
    encoded_text = llama_encoder.encode(sample_text)

    for i in range(len(encoded_text)):
        print(encoded_text[i])
    pickle.dump(encoded_text[1], open('./data/dataset/naive_language.pkl', 'wb'))
    
