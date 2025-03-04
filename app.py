import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import torch
from transformers import pipeline
from transformers import AutoTokenizer


class InferlessPythonModel:
    def initialize(self):
        model_id = 'tiiuae/falcon-7b-instruct'
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.generator = pipeline(
            'text-generation',
            model=model_id,
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
