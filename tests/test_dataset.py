from midtools import Dataset

def test_dataset_attributes():

    # get the command line arguments
    data = Dataset('./tests/setup_test_data.yml')
    assert 0

    print(f"\n{' Starting Analysis ':-^50}")
    print(f"Analyzing {data.ntrains} trains of {data.datdir}")

    print(f"Found {data.train_indices.size} complete trains")
    print(f"Finished: elapsed time: {elapsed_time/60:.2f}min")
    print(f"Results saved under {data.file_name}")
