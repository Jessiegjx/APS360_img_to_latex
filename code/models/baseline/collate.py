import torch
import torch.nn.functional as F
from models.transformer.im2latex_dataset import Im2LatexDataset


def collate_fn(batch):

    images, captions = zip(*batch)

    # --------------------
    # image padding
    # --------------------

    max_h = max(
        img.shape[1] for img in images
    )

    max_w = max(
        img.shape[2] for img in images
    )


    padded_images = []

    for img in images:

        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]

        img = F.pad(
            img,
            (
                0,
                pad_w,
                0,
                pad_h
            ),
            value=1
        )

        padded_images.append(img)


    images = torch.stack(
        padded_images
    )


    # --------------------
    # caption padding
    # --------------------

    max_len = max(
        len(c) for c in captions
    )


    padded_caps = []
    lengths = []


    for c in captions:

        lengths.append(len(c))

        pad = torch.zeros(
            max_len-len(c),
            dtype=torch.long
        )

        padded_caps.append(
            torch.cat([c,pad])
        )


    captions = torch.stack(
        padded_caps
    )

    lengths = torch.tensor(lengths)


    return images, captions, lengths