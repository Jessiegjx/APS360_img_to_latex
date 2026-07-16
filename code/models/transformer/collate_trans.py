import torch
import torch.nn.functional as F


def collate_transformer(batch):

    images, captions = zip(*batch)


    # ======================
    # Image padding
    # ======================

    max_h = max(
        img.shape[1] for img in images
    )

    max_w = max(
        img.shape[2] for img in images
    )


    padded_images=[]


    for img in images:

        pad_h=max_h-img.shape[1]
        pad_w=max_w-img.shape[2]


        img=F.pad(
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



    images=torch.stack(
        padded_images
    )



    # ======================
    # Caption padding
    # ======================


    pad_idx=0   # <PAD> in your tokenizer


    max_len=max(
        len(c) for c in captions
    )


    padded_caps=[]


    for c in captions:

        pad=torch.full(
            (max_len-len(c),),
            pad_idx,
            dtype=torch.long
        )


        padded_caps.append(
            torch.cat(
                [c,pad]
            )
        )


    captions=torch.stack(
        padded_caps
    )



    # ======================
    # Transformer padding mask
    # ======================

    # True means "ignore this token"

    padding_mask = captions == pad_idx



    return (
        images,
        captions,
        padding_mask
    )