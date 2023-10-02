import torch


def topk_mean(m: torch.Tensor, k: int):
    """
    Computes terms for the CSLS criterion in advance
    (average distance to the k-closest neighbors)
    """
    n = m.size()[0]
    ans = torch.zeros((n,), dtype=m.dtype, device=m.device)
    if k <= 0:
        return ans
    m = m.clone()
    ind0 = torch.arange(0, n, 1, dtype=torch.long, device=m.device)
    ind1 = torch.arange(0, n, 1, dtype=torch.long, device=m.device)
    minimum = torch.min(m)
    for i in range(k):
        ind1 = torch.argmax(m, dim=1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def compute_average_similarity(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    device="cpu",
    batch_size=800,
    mean_centered=False,
    mean_x=None,
    mean_y=None,
    with_random=True,
):
    x = src_emb.to(device)
    y = tgt_emb.to(device)
    if mean_centered:
        x -= mean_x.to(device) if mean_x is not None else torch.mean(x, axis=0, keepdim=True)
        y -= mean_y.to(device) if mean_y is not None else torch.mean(y, axis=0, keepdim=True)
    x /= torch.norm(x, p=2, dim=-1, keepdim=True)
    y /= torch.norm(y, p=2, dim=-1, keepdim=True)

    pair_similarities = torch.zeros((x.size()[0],), dtype=x.dtype, device=device)
    if with_random:
        random_similarities = torch.zeros((x.size()[0],), dtype=x.dtype, device=device)

    for i in range(0, x.size()[0], batch_size):
        j = min(i + batch_size, x.size()[0])

        pair_similarities[i:j] = torch.matmul(x[i:j], y[i:j].T).diagonal()

        if with_random:
            ids = torch.randint(1, y.size()[0], size=(j - i,))
            ids += torch.arange(i, j)
            ids %= y.size()[0]

            random_similarities[i:j] = torch.matmul(x[i:j], y[ids].T).diagonal()

    res = pair_similarities.to("cpu")
    if with_random:
        res = (res, random_similarities.to("cpu"))
    return res


def evaluate_alignment_with_cosim(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    device="cpu",
    batch_size=800,
    mean_centered=False,
    mean_x=None,
    mean_y=None,
    strong_alignment=False,
    ids=None,
):
    x = src_emb.to(device)
    y = tgt_emb.to(device)
    ids = ids or torch.arange(0, x.size()[0], 1, dtype=torch.long, device=device)

    if mean_centered:
        x -= mean_x.to(device) if mean_x is not None else torch.mean(x, axis=0, keepdim=True)
        y -= mean_y.to(device) if mean_y is not None else torch.mean(y, axis=0, keepdim=True)
    x /= torch.norm(x, p=2, dim=-1, keepdim=True)
    y /= torch.norm(y, p=2, dim=-1, keepdim=True)

    if strong_alignment:
        y = torch.cat((y, x), dim=0)

    batch = torch.zeros((batch_size, y.size()[0]), device=device)
    predictions = torch.zeros((x.size()[0],), dtype=torch.long, device=device)

    for i in range(0, x.size()[0], batch_size):
        j = min(i + batch_size, x.size()[0])
        torch.matmul(x[i:j], y.T, out=batch[: j - i])
        if strong_alignment:
            batch[range(j - i), range(i + tgt_emb.shape[0], j + tgt_emb.shape[0])] = 0
        predictions[i:j] = torch.argmax(batch[: j - i], axis=1)

    res = torch.count_nonzero(
        (predictions - torch.arange(0, x.size()[0], 1, dtype=torch.long, device=device))[ids]
    )
    return 1 - int(res.cpu()) / ids.size()[0]


def evaluate_alignment_with_cosim_and_knn(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    device="cpu",
    batch_size=800,
    csls_k=0,
    mean_centered=False,
    mean_x=None,
    mean_y=None,
    strong_alignment=0.0,
):
    """
    Evaluate the alignment between two list of vectors (i-th of one list is supposed to be
    the closest to the i-th of the other one) with cosine similarity (potentially with CSLS criterion)
    Note: the strong_alignment parameter allows to mix element from both languages into the search space
    """
    x = src_emb.to(device)
    y = tgt_emb.to(device)
    split_index = int(strong_alignment * x.size()[0])
    if mean_centered:
        x -= mean_x.to(device) if mean_x is not None else torch.mean(x, axis=0, keepdim=True)
        y -= mean_y.to(device) if mean_y is not None else torch.mean(y, axis=0, keepdim=True)
    x /= torch.norm(x, p=2, dim=-1, keepdim=True)
    y /= torch.norm(y, p=2, dim=-1, keepdim=True)

    batch = torch.zeros((batch_size, x.size()[0]), dtype=x.dtype, device=device)
    knn_sim = torch.zeros((y.size()[0],), dtype=y.dtype, device=device)

    if csls_k > 0:
        for i in range(0, y.size()[0], batch_size):
            j = min(i + batch_size, y.size()[0])
            torch.matmul(y[i:j], x.T, out=batch[: j - i])
            knn_sim[i:j] = topk_mean(batch[: j - i], k=csls_k)

    batch = torch.zeros((batch_size, x.size()[0]), dtype=x.dtype, device=device)
    knn_sim_x = torch.zeros((y.size()[0],), dtype=y.dtype, device=device)

    # computing mixed KNN sim for strong alignment
    if split_index > 0:
        if csls_k > 0:
            for i in range(0, split_index, batch_size):
                j = min(i + batch_size, y.size()[0])
                torch.matmul(x[i:j], x.T, out=batch[: j - i])
                knn_sim_x[i:j] = topk_mean(batch[: j - i], k=csls_k)
    knn_sim_x[split_index:] = knn_sim[split_index:]

    batch = torch.zeros((batch_size, y.size()[0]), device=device)
    predictions = torch.zeros((x.size()[0],), dtype=torch.long, device=device)

    for i in range(0, x.size()[0], batch_size):
        j = min(i + batch_size, x.size()[0])
        if strong_alignment == 1.0:
            torch.matmul(x[i:j], x.T, out=batch[: j - i])
            batch[range(j - i), range(i, j)] = torch.matmul(x[i:j], y[i:j].T).diagonal()
        elif split_index > 0:
            torch.matmul(x[i:j], x[:split_index].T, out=batch[: j - i, :split_index])
            torch.matmul(x[i:j], y[split_index:].T, out=batch[: j - i, split_index:])
            batch[range(j - i), range(i, j)] = torch.matmul(x[i:j], y[i:j].T).diagonal()
        else:
            torch.matmul(x[i:j], y.T, out=batch[: j - i])
        batch[: j - i] -= knn_sim_x / 2
        batch[range(j - i), range(i, j)] += knn_sim_x[range(i, j)] / 2
        batch[range(j - i), range(i, j)] -= knn_sim[range(i, j)] / 2
        predictions[i:j] = torch.argmax(batch[: j - i], axis=1)

    res = torch.count_nonzero(
        predictions - torch.arange(0, x.size()[0], 1, dtype=torch.long, device=device)
    )
    return 1 - int(res.cpu()) / x.size()[0]


def evaluate_alignment_with_l2(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    device="cpu",
    batch_size=800,
    mean_centered=False,
    strong_alignment=0.0,
):
    """
    Evaluate the alignment between two list of vectors (i-th of one list is supposed to be
    the closest to the i-th of the other one) with l2-distance
    Note: the strong_alignment parameter allows to mix element from both languages into the search space
    """
    x = src_emb.to(device)
    y = tgt_emb.to(device)
    split_index = int(strong_alignment * x.size()[0])

    if mean_centered:
        x -= torch.mean(x, axis=0, keepdim=True)
        y -= torch.mean(y, axis=0, keepdim=True)

    batch = torch.zeros((batch_size, y.size()[0]), device=device)
    predictions = torch.zeros((x.size()[0],), dtype=torch.long, device=device)

    for i in range(0, x.size()[0], batch_size):
        j = min(i + batch_size, x.size()[0])
        if strong_alignment == 1.0:
            batch[: j - i] = torch.cdist(x[i:j], x)
            batch[range(j - i), range(i, j)] = torch.pow(
                torch.sum(torch.pow(x[i:j] - y[i:j], 2), axis=1), 0.5
            )
        elif split_index > 0:
            # This can surely be optimized
            batch[: j - i, :split_index] = torch.cdist(x[i:j], x[:split_index])
            batch[: j - i, split_index:] = torch.cdist(x[i:j], y[split_index:])
            batch[range(j - i), range(i, j)] = torch.pow(
                torch.sum(torch.pow(x[i:j] - y[i:j], 2), axis=1), 0.5
            )
        else:
            batch[: j - i] = torch.cdist(x[i:j], y)
        predictions[i:j] = torch.argmin(batch[: j - i], axis=1)

    res = torch.count_nonzero(
        predictions - torch.arange(0, x.size()[0], 1, dtype=torch.long, device=device)
    )
    return 1 - int(res.cpu()) / x.size()[0]
