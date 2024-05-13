import os
import json
import argparse
import psutil
import time
from typing import Dict


import torch
import transformers
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import login


def process_pdf_file(file_path: str, pipeline, save_path: str) -> None:
    """
    Process a single PDF file.

    Args:
        file_path (str): Path to the PDF file.
        pipeline (transformers.pipeline): Liama3 pipeline from hugging face.
        save_path (str): Path to save the resulting JSON files.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    messages = [
        {
            "role": "system",
            "content": "You're a chatbot designed to help in finding tables within PDFs.",
        },
        {
            "role": "assistant",
            "content": documents[1].page_content,
        },
        {
            "role": "user",
            "content": "give me complete panel summary table in json format, please make sure to give me a correct json format, also Name all the tables panelـsummary.",
        },
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    text = outputs[0]["generated_text"][len(prompt) :]

    start_index = text.find("json") + len("json") + 1
    end_index = text.find("```", start_index)
    json_str = text[start_index:end_index]

    try:
        data = json.loads(json_str)

        json_save_path = os.path.join(
            save_path, os.path.basename(file_path).replace(".pdf", "") + ".json"
        )
        with open(json_save_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error in loading json string: {e}")
        # Open the file in write mode
        txt_save_path = os.path.join(
            save_path, os.path.basename(file_path).replace(".pdf", "") + ".txt"
        )
        with open(txt_save_path, "w") as file:
            # Write the string to the file
            file.write(json_str)


def measure_performance() -> Dict[str, float]:
    """
    Measure CPU and memory usage.

    Returns:
        Dict[str, float]: Dictionary containing CPU and memory usage.
    """
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    return {"cpu_usage": cpu_usage, "memory_usage": memory_usage}


def main(conf: Dict) -> None:

    folder_path = conf["folder_path"]
    save_path = conf["save_path"]
    token = conf["token"]

    login(token=token)

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        #     max_memory=max_memory_mapping,
        #     device_map="cuda",
    )

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    os.makedirs(save_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            start_time = time.time()

            process_pdf_file(file_path, pipeline, save_path)

            end_time = time.time()
            execution_time = end_time - start_time

            performance_metrics = measure_performance()

            print(
                f"Iteration completed. CPU Usage: {performance_metrics['cpu_usage']}%, Memory Usage: {performance_metrics['memory_usage']}%, Execution Time: {execution_time} seconds"
            )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing PDF files.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        default="panel_summary",
        help="Path to save the resulting JSON files.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="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",
        required=True,
        help="Hugging face login token.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    main(conf)
