

def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)

    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m]


def compute_cer(preds, refs):
    total_edits = 0
    total_chars = 0

    for pred, ref in zip(preds, refs):
        total_edits += levenshtein_distance(pred, ref)
        total_chars += len(ref)

    if total_chars == 0:
        return 0.0

    return total_edits / total_chars

def greedy_decode(log_probs, blank_id=0):
    # log_probs: [T, B, C]
    pred = log_probs.argmax(dim=2).transpose(0, 1)  # [B, T]
    texts = []

    for seq in pred:
        seq = seq.tolist()
        decoded = []
        prev = None
        for token in seq:
            if token != blank_id and token != prev:
                decoded.append(str(token - 1))  # because 1..10 -> '0'..'9'
            prev = token
        texts.append("".join(decoded))
    return texts