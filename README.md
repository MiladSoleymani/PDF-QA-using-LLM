# PDF Q&A Using LLM

A Python application that processes PDF documents and extracts structured information using Large Language Models (LLMs). This tool supports both OpenAI's GPT and Meta's LLaMA models for intelligent document analysis and question answering.

## Overview

This project provides an automated solution for extracting and analyzing information from PDF documents, particularly focused on medical reports and test results. It processes PDFs and generates structured outputs including patient information, test results, and panel summaries.

## Features

- **Dual LLM Support**: Choose between OpenAI GPT or LLaMA models for processing
- **PDF Processing**: Automated extraction and parsing of PDF content
- **Structured Output**: Generates organized CSV/JSON files with extracted information
- **Batch Processing**: Process multiple PDF files in a directory
- **Resource Monitoring**: Built-in memory usage tracking during processing

## Prerequisites

- Python 3.8 or higher
- API access (OpenAI API key or Hugging Face token)
- Sufficient system memory for LLaMA model (if using local inference)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PDF-QA-using-LLM.git
cd PDF-QA-using-LLM
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using OpenAI GPT

Process PDFs using OpenAI's GPT models:

```bash
python GPT_qa.py --folder_path <input_directory> --save_path <output_directory> --api_key <your_openai_api_key>
```

**Parameters:**
- `--folder_path`: Directory containing PDF files to process
- `--save_path`: Directory where output files will be saved
- `--api_key`: Your OpenAI API key

**Example:**
```bash
python GPT_qa.py --folder_path ./pdfs --save_path ./results --api_key sk-your-api-key
```

### Using LLaMA

Process PDFs using Meta's LLaMA model:

```bash
python Liama_qa.py --folder_path <input_directory> --save_path <output_directory> --token <your_huggingface_token>
```

**Parameters:**
- `--folder_path`: Directory containing PDF files to process
- `--save_path`: Directory where output files will be saved
- `--token`: Your Hugging Face access token

**Example:**
```bash
python Liama_qa.py --folder_path ./pdfs --save_path ./results --token hf_your_token
```

## Output Format

The scripts generate structured output files for each processed PDF:

- `<filename>_patient_info.csv`: Extracted patient information
- `<filename>_test_results.csv`: Test results and measurements
- `<filename>_panel_summary.csv`: Summary of test panels

## Project Structure

```
PDF-QA-using-LLM/
├── GPT_qa.py           # Main script for GPT processing
├── Liama_qa.py         # Main script for LLaMA processing
├── helpers/            # Helper modules
│   ├── GPT.py         # GPT-specific functions
│   ├── LIAMA.py       # LLaMA-specific functions
│   └── __init__.py
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

- `psutil`: System resource monitoring
- `torch`: PyTorch framework (for LLaMA)
- `transformers`: Hugging Face transformers library
- `accelerate`: Training acceleration library
- `langchain`: Document processing framework
- `pypdf`: PDF parsing library
- `openai`: OpenAI API client

## Notes

- Ensure you have sufficient API credits when using OpenAI GPT
- LLaMA processing requires significant GPU memory (recommended: 16GB+ VRAM)
- Processing time varies based on PDF size and complexity
- The tool is optimized for medical reports but can be adapted for other document types

## License

This project is open source. Please check the license file for more details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
