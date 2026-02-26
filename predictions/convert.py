import pandas as pd

df = pd.read_csv("old_para-dev-output.csv", sep=",")

# Convert 0/1 → no/yes
df[" A"] = df[" A"].map({
    1: " 8505",
    0: " 3919"
})

df.to_csv("para-dev-output.csv", index=False)