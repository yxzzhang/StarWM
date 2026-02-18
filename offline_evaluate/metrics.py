import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import math

def calc_smape(pred, gt):
    if pred == 0 and gt == 0:
        return 0.0
    epsilon = 1e-10
    return abs(pred - gt) / ((abs(pred) + abs(gt)) / 2.0 + epsilon)

def calc_mae(pred, gt):
    return abs(pred - gt)

def calc_f1(tp, fp, fn):
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calc_precision(tp, fp):
    if tp + fp == 0: return 0.0
    return tp / (tp + fp)

def calc_recall(tp, fn):
    if tp + fn == 0: return 0.0
    return tp / (tp + fn)

def compute_id_hungarian_matching(gt_units, pred_units, dist_threshold=10.0):
    """
    Match strategy:
    1. Exact ID Match (with Type Assert)
    2. Hungarian Match on residuals (Distance < threshold AND Type match)
    """
    matches = []
    
    # 1. ID Match
    # Map by ID
    gt_map = {u.id: u for u in gt_units if u.id is not None}
    pred_map = {u.id: u for u in pred_units if u.id is not None}
    
    # Identify common IDs
    common_ids = set(gt_map.keys()) & set(pred_map.keys())
    
    matched_gt_ids = set()
    matched_pred_ids = set()
    
    for uid in common_ids:
        u_gt = gt_map[uid]
        u_pred = pred_map[uid]
        
        # Assert type
        if u_gt.type == u_pred.type:
            matches.append((u_gt, u_pred))
            matched_gt_ids.add(uid)
            matched_pred_ids.add(uid)
            
    # Residuals
    gt_resid = [u for u in gt_units if u.id is None or u.id not in matched_gt_ids]
    pred_resid = [u for u in pred_units if u.id is None or u.id not in matched_pred_ids]
    
    if not gt_resid or not pred_resid:
        return matches, gt_resid, pred_resid
        
    # 2. Hungarian on Residuals
    # Filter by type to avoid huge matrix? No, type penalty is better in cost matrix.
    
    m = len(gt_resid)
    n = len(pred_resid)
    cost_matrix = np.zeros((m, n))
    
    LARGE_COST = 1e6
    
    for i, g in enumerate(gt_resid):
        for j, p in enumerate(pred_resid):
            if g.type != p.type:
                cost_matrix[i, j] = LARGE_COST
            else:
                d = np.linalg.norm(np.array(g.pos) - np.array(p.pos))
                if d > dist_threshold:
                    cost_matrix[i, j] = LARGE_COST
                else:
                    cost_matrix[i, j] = d
                    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched_resid_gt_indices = set()
    matched_resid_pred_indices = set()
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < LARGE_COST:
            matches.append((gt_resid[r], pred_resid[c]))
            matched_resid_gt_indices.add(r)
            matched_resid_pred_indices.add(c)
            
    final_unmatched_gt = [u for i, u in enumerate(gt_resid) if i not in matched_resid_gt_indices]
    final_unmatched_pred = [u for i, u in enumerate(pred_resid) if i not in matched_resid_pred_indices]
    
    return matches, final_unmatched_gt, final_unmatched_pred

def solve_awd(gt_units, pred_units, map_size=64, type_penalty=1e6):
    """
    Calculates AWD Cost and returns matching details.
    
    Returns:
        total_cost (float): Normalized Average AWD Cost
        matches (list): List of (gt_unit, pred_unit) pairs from D_match
        unmatched_gt (list): List of gt_units that were Missed (matched to Dummy)
        unmatched_pred (list): List of pred_units that were Hallucinations (matched to Dummy)
    """
    m = len(gt_units)
    n = len(pred_units)
    
    if m == 0 and n == 0:
        return 0.0, [], [], []
        
    # tau = map_size * math.sqrt(2) # Penalty Threshold
    tau = map_size * math.sqrt(2) # Penalty Threshold
    
    # If one side is empty
    if m == 0:
        # All Preds are Hallucinations
        cost = n * tau
        norm_cost = cost / n # (M+N) = N
        return norm_cost, [], [], list(pred_units)
        
    if n == 0:
        # All GTs are Misses
        cost = m * tau
        norm_cost = cost / m # (M+N) = M
        return norm_cost, [], list(gt_units), []

    gt_coords = np.array([u.pos for u in gt_units])
    pred_coords = np.array([u.pos for u in pred_units])
    
    # 1. Distance Matrix (M x N)
    dist_mat = cdist(gt_coords, pred_coords, metric='euclidean')
    
    # Type Constraint
    for i, g in enumerate(gt_units):
        for j, p in enumerate(pred_units):
            if g.type != p.type:
                dist_mat[i, j] = type_penalty

    # 2. Augmented Matrix
    INF_COST = type_penalty
    
    # Top Left: Match Cost
    top_left = dist_mat
    
    # Top Right: Miss Cost (GT -> Dummy)
    # M x M diagonal tau
    top_right = np.full((m, m), INF_COST)
    np.fill_diagonal(top_right, tau)
    
    # Bottom Left: Hallucination Cost (Dummy -> Pred)
    # N x N diagonal tau
    bottom_left = np.full((n, n), INF_COST)
    np.fill_diagonal(bottom_left, tau)
    
    # Bottom Right: Dummy -> Dummy (Zero Cost)
    # N x M
    bottom_right = np.zeros((n, m))
    
    # Construct Full Matrix (M+N) x (M+N)
    # Be careful with dimensions:
    # Rows: M (GT) + N (Dummy for Hallucination source) = M + N
    # Cols: N (Pred) + M (Dummy for Miss target) = N + M
    #
    # Wait, the README structure:
    # [ D_match (MxN)   D_miss (MxM) ]
    # [ D_hall (NxN)    D_dummy (NxM) ]
    #
    # Rows: First M are GT units. Next N are Dummies (representing "hallucinated units source").
    # Cols: First N are Pred units. Next M are Dummies (representing "missed units sink").
    
    top_block = np.hstack([top_left, top_right])      # (M, N+M)
    bottom_block = np.hstack([bottom_left, bottom_right]) # (N, N+M)
    cost_matrix = np.vstack([top_block, bottom_block])    # (M+N, N+M)
    
    # 3. Solve
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 4. Interpret Results
    total_cost = 0.0
    matches = []
    unmatched_gt = []
    unmatched_pred = []
    
    # Sets to track covered indices
    gt_covered = set()
    pred_covered = set()
    
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        
        # Skip INF costs (should not happen in optimal solution unless forced)
        if cost >= INF_COST: 
            # In practice, if forced, we treat it as max penalty?
            # But with Dummy nodes available at cost tau, solver shouldn't pick INF.
            raise ValueError(f"should not be INF: {cost}")
            
        # Case 1: GT(r) -> Pred(c) [Top Left]
        if r < m and c < n:
            matches.append((gt_units[r], pred_units[c]))
            gt_covered.add(r)
            pred_covered.add(c)
            total_cost += cost
            
        # Case 2: GT(r) -> Dummy(c) [Top Right] -> Miss
        elif r < m and c >= n:
            # GT r matched to Dummy (c-n).
            # This is a Miss.
            unmatched_gt.append(gt_units[r])
            total_cost += cost # Should be tau
            
        # Case 3: Dummy(r) -> Pred(c) [Bottom Left] -> Hallucination
        elif r >= m and c < n:
            # Dummy (r-m) matched to Pred c.
            # This is a Hallucination.
            unmatched_pred.append(pred_units[c])
            total_cost += cost # Should be tau
            
        # Case 4: Dummy -> Dummy [Bottom Right] -> Zero Cost
        elif r >= m and c >= n:
            pass
            
    # Calculate Average AWD
    avg_cost = total_cost / (m + n)
    
    return avg_cost, matches, unmatched_gt, unmatched_pred
