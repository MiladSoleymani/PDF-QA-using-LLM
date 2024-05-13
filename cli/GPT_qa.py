import os
import json
import argparse
import psutil
import time
from typing import Dict

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader


def process_pdf_file(file_path: str, api_key: str, save_path: str) -> None:
    """
    Process a single PDF file.

    Args:
        file_path (str): Path to the PDF file.
        api_key (str): OpenAI API key.
        save_path (str): Path to save the resulting JSON files.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    client = OpenAI(api_key=api_key)

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

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4-turbo",
    )

    text = chat_completion.choices[0].message.content

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
    api_key = conf["api_key"]

    os.makedirs(save_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            start_time = time.time()

            process_pdf_file(file_path, api_key, save_path)

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
        default="panel_summary",
        help="Path to save the resulting JSON files.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    main(conf)
