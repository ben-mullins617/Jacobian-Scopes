import os

# Use shared memory for Hugging Face cache to avoid workspace disk space limits.
os.environ["HF_HOME"] = "/dev/shm/.hf_home"
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

import torch
import argparse
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_ID = "google/gemma-4-31B-it"
PROMPT = "Write a short joke about saving RAM."


def test_model(model, processor):

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    # Parse output
    print(processor.parse_response(response))


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="/dev/shm/hf_offload",
        offload_state_dict=True,
    )
    model.eval()
    messages = [
        {"role": "user", "content": PROMPT},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=16,
            early_stopping=True,
        )

    generated_ids = sequences[0, inputs["input_ids"].shape[-1] :]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    torch.save(
        {
            "model_id": MODEL_ID,
            "token_ids": sequences[0].detach().cpu(),
            "full_sequence": True,
        },
        "residual_stream.pt",
    )

    print(f"Prompt: {PROMPT}")
    print(f"Response: {response}")
    print(f"Saved residual-stream tensors to hidden_states/residual_stream.pt")