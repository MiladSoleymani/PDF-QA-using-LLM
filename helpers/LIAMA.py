import json

import pandas as pd


def patient_info_table(page_content: str, pipeline, save_path: str) -> None:

    messages = [
        {"role": "system", "content": "You're a chatbot designed to help in finding tables within PDFs.",},
        {"role": "assistant", "content": page_content,},
        {"role": "user", "content": "give me the first table completely in json format, Name all the tables patient-info",},
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
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    text = outputs[0]["generated_text"][len(prompt) :]

    start_index = text.find("```") + len("```") + 1
    end_index = text.find("```", start_index)
    json_str = text[start_index:end_index]

    try:
        json_data = json.loads(json_str)

        df = pd.DataFrame.from_dict(json_data["patient-info"][0], orient="index")
        df.to_csv(save_path)

    except Exception as e:
        print(f"Error in loading json string: {e}")
        with open(save_path.replace(".csv", ".txt"), "w") as file:
            # Write the string to the file
            file.write(json_str)


def test_results_table(page_content: str, pipeline, save_path: str) -> None:

    messages = [
        {"role": "system", "content": "You're a chatbot designed to help in finding tables within PDFs."},
        {"role": "assistant", "content": page_content},
        {"role": "user", "content": "give me Test Results table in json format, Name all the tables Test-Results, table contains ASSUMED ORIGIN, GENE, VARIANT(this contain two rows and you should put them in one string), VRF(%), CLINICAL SIGNIFICANCE IN MPN"},
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
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    text = outputs[0]["generated_text"][len(prompt) :]

    start_index = text.find("```") + len("```") + 1
    end_index = text.find("```", start_index)
    json_str = text[start_index:end_index]

    try:
        json_data = json.loads(json_str)

        result = {}

        for idx, data in enumerate(json_data["Test-Results"]):
            result[idx] = data


        df = pd.DataFrame.from_dict(result, orient="index")
        df.to_csv(save_path)

    except Exception as e:
        print(f"Error in loading json string: {e}")
        with open(save_path.replace(".csv", ".txt"), "w") as file:
            # Write the string to the file
            file.write(json_str)



def panel_summary_table(page_content: str, pipeline, save_path: str) -> None:
    
    messages = [
        {"role": "system", "content": "You're a chatbot designed to help in finding tables within PDFs."},
        {"role": "assistant", "content": page_content},
        {"role": "user", "content": "give me the pannel summary table completely in json format, Name all the tables panel-summary"},
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
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    text = outputs[0]["generated_text"][len(prompt) :]

    start_index = text.find("```") + len("```") + 1
    end_index = text.find("```", start_index)
    json_str = text[start_index:end_index]

    try:
        json_data = json.loads(json_str)

        result = {}

        for idx, data in enumerate(json_data["panel-summary"]):
            result[idx] = data

        df = pd.DataFrame.from_dict(result, orient="index")
        df.to_csv(save_path)

    except Exception as e:
        print(f"Error in loading json string: {e}")
        with open(save_path.replace(".csv", ".txt"), "w") as file:
            # Write the string to the file
            file.write(json_str)