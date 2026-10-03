# this is responsible for downloading/loading the BANKING77 dataset and labelling from Hugging Face 
# and persisting it to disk for later use into data/raw/.

# it should not tokenize text, train any models, encode labels, 
# remove duplicates, normalizetext, split the dataset or calculate metrics

#using hugging face datasets to obtain BANKING77 and save the untouched source data locally.


# Import necessary libraries
from pathlib import Path

from datasets import load_dataset


# Define the Banking77Ingestion class
class Banking77Ingestion:
    """Download and persist the raw BANKING77 dataset."""

    DATASET_REVISION = "830c4f2be6949546b11fe83fbc50993348a2bccd"

    def __init__(self, output_dir: str = "data/raw") -> None:
        self.output_dir = Path(output_dir)

    def download(self):
        """Download BANKING77 from the pinned Hugging Face revision."""

        base_url = (
            "https://huggingface.co/datasets/PolyAI/banking77/"
            f"resolve/{self.DATASET_REVISION}/data"
        )

        dataset = load_dataset(
            "parquet",
            data_files={
                "train": (
                    f"{base_url}/train-00000-of-00001.parquet"
                ),
                "test": (
                    f"{base_url}/test-00000-of-00001.parquet"
                ),
            },
        )

        return dataset

    def save(self, dataset) -> None:
        """Save each dataset split to disk."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split in dataset:
            split_path = self.output_dir / split
            dataset[split].save_to_disk(str(split_path))

    def run(self) -> None:
        """Execute the complete ingestion process."""

        dataset = self.download()
        self.save(dataset)


# Run the ingestion process if the script is executed directly
if __name__ == "__main__":
    ingestor = Banking77Ingestion()
    ingestor.run()