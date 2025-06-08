# run_llm.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def main():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    # model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    # model_id = "tiiuae/falcon-7b-instruct"
    # model_id = "databricks/dolly-v2-12b"

    # 4-bit 양자화 설정
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    # )

    # 토크나이저 & 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     use_auth_token=True,    
    # )

    # bitsandbytes 양자화 제거 → FP16-only
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=True,
    )

    # 파이프라인 생성
    chat = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
    )

    # 대화 루프
    while True:
        user_input = input("\n[You] ")
        if user_input.lower() in ("exit", "quit"):
            break
        response = chat(user_input)[0]["generated_text"]
        print("\n[LLM]", response.strip())

if __name__ == "__main__":
    main()
