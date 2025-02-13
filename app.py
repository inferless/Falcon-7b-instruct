import torch
from transformers import pipeline
from transformers import AutoTokenizer



class InferlessPythonModel:
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b-instruct')
        self.generator = pipeline(
            'text-generation',
            model='tiiuae/falcon-7b-instruct',
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device=0,
        )

    def infer(self, inputs):
        pipeline_output = self.generator(
            inputs['prompt'],
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = pipeline_output[0]['generated_text']
        return {"generated_text": generated_text}

    def finalize(self):
        self.pipe = None
