import os
import torch
import argparse
from datasets import load_dataset
from transfomers import AutoProcessor, AutoModelForMultimodalLM


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
        dtype="auto",
        device_map="auto"
    )
    model.eval()
    messages = [
            {"role": "user", "content": PROMPT},
        ]
    
    # Process input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=False,
        )

        # Re-run the completed sequence once so hidden_states contains one
        # aligned tensor per residual-stream point for every token.
        traced_inputs = {
            "input_ids": sequences,
            "attention_mask": torch.ones_like(sequences),
        }
        outputs = model(
            **traced_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = getattr(outputs, "hidden_states", None)
        
    generated_ids = sequences[0, inputs["input_ids"].shape[-1] :]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    states_to_save: list[torch.Tensor] = []
    for state in hidden_states:
        selected = state[0]
        states_to_save.append(selected.detach().to(device="cpu", dtype=torch.float16))

    torch.save(
        {
            "model_id": MODEL_ID,
            "token_ids": sequences[0].detach().cpu(),
            "states": states_to_save,
            "full_sequence": True,
        },
        "residual_stream.pt",
    )

    print(f"Prompt: {PROMPT}")
    print(f"Response: {response}")
    print(f"Saved {len(states_to_save)} residual-stream tensors to residual_stream.pt")