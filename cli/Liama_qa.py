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

import pandas as pd

from LIAMA import patient_info_table, test_results_table, panel_summary_table


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

    patient_info_table(documents[0].page_content, pipeline, os.path.join(
            save_path, os.path.basename(file_path).replace(".pdf", "") + "_patient_info.csv"
        ))

    test_results_table(documents[0].page_content, pipeline, os.path.join(
            save_path, os.path.basename(file_path).replace(".pdf", "") + "_test_results_table.csv"
        ))
    
    panel_summary_table(documents[1].page_content, pipeline, os.path.join(
            save_path, os.path.basename(file_path).replace(".pdf", "") + "_panel_summary_table.csv"
        ))

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
