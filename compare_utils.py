import torch

def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float = 0) -> bool:
    """Compare based or not in a threshold, if threshold is none then it is equal comparison"""
    if threshold > 0:
        return bool(torch.all(torch.le(torch.abs(torch.subtract(rhs, lhs)), threshold)))
    else:
        return bool(torch.equal(rhs, lhs))


def get_top_k_labels(tensor: torch.tensor, top_k: int) -> torch.tensor:
    proba = torch.nn.functional.softmax(tensor, dim=1)
    return torch.topk(proba, k=top_k).indices.squeeze(0)


def compare_classification(
    output_tsr: torch.tensor, golden_tsr: torch.tensor, top_k: int, logger=None
) -> int:
    errors = {}
    output_tsr, golden_tsr = output_tsr.to("cpu"), golden_tsr.to("cpu")

    # tensor comparison
    if not equal(output_tsr, golden_tsr, threshold=1e-4):
        for i, (output, golden) in enumerate(zip(output_tsr, golden_tsr)):
            if not equal(output, golden):
                errors[i] = (1, 0)

    # top k comparison to check if classification has changed
    output_topk = get_top_k_labels(output_tsr, top_k)
    golden_topk = get_top_k_labels(golden_tsr, top_k)
    if equal(output_topk, golden_topk) is False:
        for i, (tpk_found, tpk_gold) in enumerate(zip(output_topk, golden_topk)):
            if tpk_found != tpk_gold:
                if i in errors:
                    output, _ = errors[i]
                    errors[i] = (output, 1)
                else:
                    errors[i] = (0, 1)

    return errors