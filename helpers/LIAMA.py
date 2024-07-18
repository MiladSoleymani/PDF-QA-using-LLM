import json
import pandas as pd


def extract_table(
    page_content: str,
    pipeline,
    save_path: str,
    table_name: str,
    user_message: str,
    max_new_tokens: int,
) -> None:
    messages = [
        {
            "role": "system",
            "content": "You're a chatbot designed to help in finding tables within PDFs.",
        },
        {"role": "assistant", "content": page_content},
        {"role": "user", "content": user_message},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids(""),
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    text = outputs[0]["generated_text"][len(prompt) :]

    start_index = text.find("```json") + len("```json")
    end_index = text.find("```", start_index)
    json_str = text[start_index:end_index].strip()

    try:
        json_data = json.loads(json_str)
        result = {idx: data for idx, data in enumerate(json_data[table_name])}

        df = pd.DataFrame.from_dict(result, orient="index")
        df.to_csv(save_path)
    except Exception as e:
        print(f"Error in loading JSON string: {e}")
        with open(save_path.replace(".csv", ".txt"), "w") as file:
            file.write(json_str)


def patient_info_table(page_content: str, pipeline, save_path: str) -> None:
    user_message = "give me the first table completely in json format, Name all the tables patient-info"
    extract_table(page_content, pipeline, save_path, "patient-info", user_message, 2048)


def test_results_table(page_content: str, pipeline, save_path: str) -> None:
    user_message = (
        "give me Test Results table in json format, Name all the tables Test-Results, "
        "table contains ASSUMED ORIGIN, GENE, VARIANT(this contain two rows and you should put them in one string), "
        "VRF(%), CLINICAL SIGNIFICANCE IN MPN"
    )
    extract_table(page_content, pipeline, save_path, "Test-Results", user_message, 2048)


def panel_summary_table(page_content: str, pipeline, save_path: str) -> None:
    user_message = "give me the panel summary table completely in json format, Name all the tables panel-summary"
    extract_table(
        page_content, pipeline, save_path, "panel-summary", user_message, 4096
    )
