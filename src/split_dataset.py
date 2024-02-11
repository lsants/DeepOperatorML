from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def split_dataset(data, TEST_SIZE=0.2, seed=0):
    """Split dataset in training and test sets.

    Args:
        data: dataset
        TEST_SIZE (float, optional): Fraction of dataset dedicated to training the model. Defaults to 0.2.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        split (tuple): training and validation sets
    """
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(data)),
        data[:][1],
        test_size=TEST_SIZE,
        random_state=seed
    )

    # generate subset based on indices
    train_split = Subset(data, train_indices)
    test_split = Subset(data, test_indices)

    return train_split, test_split
