import torch
from transformers import AutoTokenizer, AutoModel


def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    text = "Patient presents with fever and cough. CXR shows consolidation."
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=64)

    with torch.no_grad():
        out = model(**inputs)

    # last_hidden_state shape: [batch, seq_len, hidden]
    print("ClinicalBERT last_hidden_state:",
          tuple(out.last_hidden_state.shape))
    print("Sanity check OK ✅")


if __name__ == "__main__":
    main()
