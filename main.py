from datasets import load_dataset

ds = load_dataset("pinzhenchen/wmt26-mist-sample")
# print(ds)              # shows splits (train/test/etc.) and columns

df = ds["train"].to_pandas()

for (task, source), group in df.groupby(["task", "source"]):
    print(task, source, len(group))
    # do something with `group`