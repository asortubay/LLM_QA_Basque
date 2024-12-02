import json
from datasets import load_dataset

input_filename = "eusquad_v1.0\eusquad-train-v1.0_canine-s.json"
output_filename = "eusquad_v1.0\eusquad-train-v1.0_canine-s.jsonl"

with open(input_filename) as f:
    dataset = json.load(f)

with open(output_filename, "w") as f:
    for article in dataset["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            answers = {}
            for qa in paragraph["qas"]:
                question = qa["question"]
                idx = qa["id"]
                answers["text"] = [a["text"] for a in qa["answers"]]
                answers["answer_start"] = [a["answer_start"] for a in qa["answers"]]
                f.write(
                    json.dumps(
                        {
                            "id": idx,
                            "title": title,
                            "context": context,
                            "question": question,
                            "answers": answers,
                        }
                    )
                )
                f.write("\n")