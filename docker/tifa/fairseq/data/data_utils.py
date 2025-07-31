# Simplified copy of fairseq's helper
import math
import numpy as np
import torch


def compute_mask_indices(
    shape,
    mask_prob,
    mask_length,
    device="cpu",
    min_masks=1,
):
    """
    Вернёт Tensor bool [B, T] – True там, где нужно «маскировать» токен.
    shape: Tuple[int,int]  (batch, seq_len)
    mask_prob: доля позиций под маску (0‒1)
    mask_length: длина каждого непрерывного блока
    """

    batch_size, seq_length = shape
    mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # сколько блоков в среднем на один элемент
    num_mask = max(
        min_masks,
        int(math.floor(mask_prob * seq_length / mask_length + np.random.rand())),
    )

    for i in range(batch_size):
        chosen = set()
        for _ in range(num_mask):
            start = np.random.randint(0, max(1, seq_length - mask_length))
            # избегаем сильного перекрытия
            while any(s <= start < s + mask_length for s in chosen):
                start = np.random.randint(0, max(1, seq_length - mask_length))
            chosen.add(start)
            mask[i, start : start + mask_length] = True

    return mask
