import modal
import pandas as pd

app = modal.App("inspect-data")
image = modal.Image.debian_slim().pip_install("pandas")
volume = modal.Volume.from_name("writeo-training-data")


@app.function(image=image, volumes={"/data": volume})
def inspect():
    print("Inspecting DREsS_Std...")
    try:
        df = pd.read_csv("/data/dress/DREsS_Std.tsv", sep="\t")
        print(f"Columns: {list(df.columns)}")
        for col in ["content", "organization", "language"]:
            if col in df.columns:
                print(
                    f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}"
                )
            elif col.title() in df.columns:
                c = col.title()
                print(f"{c}: min={df[c].min()}, max={df[c].max()}, mean={df[c].mean()}")
    except Exception as e:
        print(e)

    print("\nInspecting DREsS_New...")
    try:
        df = pd.read_csv("/data/dress/DREsS_New.tsv", sep="\t")
        print(f"Columns: {list(df.columns)}")
        # normalize
        df.columns = [c.lower().strip() for c in df.columns]
        for col in ["content", "organization", "language"]:
            if col in df.columns:
                print(
                    f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}"
                )
    except Exception as e:
        print(e)
