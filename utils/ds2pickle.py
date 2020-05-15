import pickle

import midl.ds.AbdomenDataset


def ds2pickle(dataset,
              save_path):

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    # Train DS
    # ds = midl.ds.AbdomenDataset("liver",
    #                             128, 128, 64,
    #                             path_image_dir="E:/Data/INFINITT/Integrated/train/img",
    #                             path_label_dir="E:/Data/INFINITT/Integrated/train/label")

    # Test DS
    ds = midl.ds.AbdomenDataset("liver",
                             128, 128, 64,
                             path_image_dir="E:/Data/INFINITT/Integrated/test/img",
                             path_label_dir="E:/Data/INFINITT/Integrated/test/label")

    ds2pickle(ds, './test_ds')
