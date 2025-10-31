import pandas as pd

# splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}
# df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])


# print(df.shape)

# df.to_csv("gsm8k_test.csv", index=False)
# print("Saved gsm8k_test.csv")


# df.drop(columns=["answer"], inplace=True)

# df.to_csv("gsm8k_test_no_answers.csv", index=False)
# print("Saved gsm8k_test_no_answers.csv")



df_diwali  = pd.read_json("hf://datasets/nlip/DIWALI/DIWALIv1.jsonl", lines=True)

print(df_diwali.shape)
df_diwali.drop(columns=["source"], inplace=True)
df_diwali.to_csv("diwali.csv", index=False)
print("Saved diwali.csv")