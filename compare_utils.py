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

############################################################################################################
# -- For Grounding Dino

# Convert from (x, y, w, h) to (x1, y1, x2, y2)
def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    box[:2] -= box[2:] / 2   # (x1, y1) = (x - w/2, y - h/2)
    box[2:] += box[:2]       # (x2, y2) = (x + w/2, y + h/2)
    return box


# Define the element-wise IoU function
def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    box1 = xywh_to_xyxy(box1.clone())
    box2 = xywh_to_xyxy(box2.clone())

    # Intersection
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    # Compute intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
        torch.clamp(inter_y2 - inter_y1, min=0)

    # Compute the areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union_area = box1_area + box2_area - inter_area

    # IoU calculation
    iou = inter_area / union_area
    return iou

def count_elements(lst):
    """ Count the number of each key in a dictionnary. """
    counts = {}
    for elem in lst:
        if elem in counts:
            counts[elem] += 1
        else:
            counts[elem] = 1
    return counts