from datasets import load_dataset

ds = load_dataset("locuslab/TOFU", "forget10")
print(ds)

for i in range(0, 400, 20):
    print(i, ds["train"][i]["answer"][:60])

ds["train"].to_json("forget10.jsonl")