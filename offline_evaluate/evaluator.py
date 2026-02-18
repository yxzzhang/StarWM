import numpy as np
from collections import defaultdict
from parser import SC2TextParser
from metrics import calc_smape, calc_mae, calc_f1, calc_precision, calc_recall, solve_awd, compute_id_hungarian_matching

class SC2WorldEvaluator:
    def __init__(self):
        self.parser = SC2TextParser()
        
    def evaluate_frame(self, gt_obs_text, pred_obs_text):
        """
        Evaluates a single frame prediction against ground truth.
        """
        gt_data = self.parser.parse(gt_obs_text)
        pred_data = self.parser.parse(pred_obs_text)
        
        if not gt_data or not pred_data:
            return {"error": "Parsing failed"}

        metrics = {}
        
        # --- 1. Macro Topology (AWD) ---
        # Group units for Macro AWD (using all available)
        
        # Self Units (Army only, exclude worker as per parser update)
        gt_self_units = [u for u in gt_data['units']]
        pred_self_units = [u for u in pred_data['units']]
        
        # Self Structures
        gt_structs = gt_data['structures']
        pred_structs = pred_data['structures']
        
        # Enemy Units
        gt_enemy_units = [u for u in gt_data['enemies'] if u.category == 'EnemyUnit']
        pred_enemy_units = [u for u in pred_data['enemies'] if u.category == 'EnemyUnit']
        
        # Enemy Structures
        gt_enemy_structs = [u for u in gt_data['enemies'] if u.category == 'EnemyStruct']
        pred_enemy_structs = [u for u in pred_data['enemies'] if u.category == 'EnemyStruct']
        
        # Snapshot Enemy Structures
        gt_snap_structs = [u for u in gt_data['enemies'] if u.category == 'SnapEnemyStruct']
        pred_snap_structs = [u for u in pred_data['enemies'] if u.category == 'SnapEnemyStruct']
        
        groups = {
            'Self_Unit': (gt_self_units, pred_self_units),
            'Self_Struct': (gt_structs, pred_structs),
            'Enemy_Unit': (gt_enemy_units, pred_enemy_units),
            'Enemy_Struct': (gt_enemy_structs, pred_enemy_structs),
            'Snap_Enemy_Struct': (gt_snap_structs, pred_snap_structs)
        }
        
        for name, (g_list, p_list) in groups.items():
            # Calculate Macro AWD
            cost, _, _, _ = solve_awd(g_list, p_list)
            metrics[f'Macro_AWD_{name}'] = cost
            
            # --- 2. Micro Recognition (ID + Hungarian) ---
            # Do NOT use AWD matches. Use ID + Hungarian.
            matches, unmatched_gt, unmatched_pred = compute_id_hungarian_matching(g_list, p_list)
            
            tp = len(matches)
            fn = len(unmatched_gt)
            fp = len(unmatched_pred)
            
            # Micro F1 Stats (Store raw counts for aggregation later if needed, or per-frame F1)
            # The prompt asks for Micro Avg (over dataset) and Macro Avg (over frames).
            # We return counts here to aggregate in run_eval.py for Micro Avg.
            metrics[f'Micro_Counts_{name}'] = {'tp': tp, 'fp': fp, 'fn': fn}
            
            # Per-frame (Macro) metrics
            # Logic: If GT is empty and Pred is empty -> 1.0
            # If GT is empty and Pred is NOT empty -> 0.0
            # If GT is NOT empty and Pred is empty -> 0.0
            
            is_gt_empty = len(g_list) == 0
            is_pred_empty = len(p_list) == 0
            
            if is_gt_empty and is_pred_empty:
                metrics[f'Frame_F1_{name}'] = 1.0
                metrics[f'Frame_Prec_{name}'] = 1.0
                metrics[f'Frame_Recall_{name}'] = 1.0
            elif is_gt_empty and not is_pred_empty:
                metrics[f'Frame_F1_{name}'] = 0.0
                metrics[f'Frame_Prec_{name}'] = 0.0
                metrics[f'Frame_Recall_{name}'] = 0.0
            elif not is_gt_empty and is_pred_empty:
                metrics[f'Frame_F1_{name}'] = 0.0
                metrics[f'Frame_Prec_{name}'] = 0.0
                metrics[f'Frame_Recall_{name}'] = 0.0
            else:
                metrics[f'Frame_F1_{name}'] = calc_f1(tp, fp, fn)
                metrics[f'Frame_Prec_{name}'] = calc_precision(tp, fp)
                metrics[f'Frame_Recall_{name}'] = calc_recall(tp, fn)
            
            # --- Attribute Accuracy (Only on Matched pairs) ---
            if matches:
                hp_diffs = []
                eng_diffs = []
                
                for g, p in matches:
                    if g.hp is not None and p.hp is not None:
                        hp_diffs.append(abs(g.hp - p.hp))
                    if g.eng is not None and p.eng is not None:
                        eng_diffs.append(abs(g.eng - p.eng))
                        
                if hp_diffs:
                    metrics[f'Frame_{name}_HP_MAE'] = np.mean(hp_diffs)
                if eng_diffs:
                    metrics[f'Frame_{name}_Energy_MAE'] = np.mean(eng_diffs)

        # --- 3. Economy & Status ---
        gt_info = gt_data['info']
        pred_info = pred_data['info']
        
        metrics['Time_Seconds'] = gt_info['time_seconds'] # For plotting
        
        metrics['Resource_Minerals_SMAPE'] = calc_smape(pred_info['minerals'], gt_info['minerals'])
        metrics['Resource_Gas_SMAPE'] = calc_smape(pred_info['gas'], gt_info['gas'])
        metrics['Resource_Minerals_Rate_SMAPE'] = calc_smape(pred_info['minerals_rate'], gt_info['minerals_rate'])
        metrics['Resource_Gas_Rate_SMAPE'] = calc_smape(pred_info['gas_rate'], gt_info['gas_rate'])
        metrics['Supply_Used_SMAPE'] = calc_smape(pred_info['supply_used'], gt_info['supply_used'])
        metrics['Supply_Cap_SMAPE'] = calc_smape(pred_info['supply_cap'], gt_info['supply_cap'])
        metrics['Econ_Workers_SMAPE'] = calc_smape(pred_info['workers_count'], gt_info['workers_count'])
        
        # Alerts & Upgrades (F1) - Evaluated only if GT is not empty
        if gt_info['alerts']:
            a_gt, a_p = gt_info['alerts'], pred_info['alerts']
            metrics['Status_Alerts_F1'] = calc_f1(len(a_gt & a_p), len(a_p - a_gt), len(a_gt - a_p))
            
        if gt_info['upgrades']:
            u_gt, u_p = gt_info['upgrades'], pred_info['upgrades']
            metrics['Tech_Upgrades_F1'] = calc_f1(len(u_gt & u_p), len(u_p - u_gt), len(u_gt - u_p))

        # --- 4. Development (Queue) ---
        gt_queue = gt_data['queue_items']
        pred_queue = pred_data['queue_items']
        
        # Exact Match Accuracy & Progress MAE
        # Strategy:
        # Match queue items based on keys.
        # Constructing: (uid, u_type, pos)
        # Production: (producer_id, pos, product) -> Need to handle duplicates?
        # User said: "Treat multiple productions as independent items... same key... separate calculation".
        # This implies we should treat the lists as multisets.
        
        # Helper to generate keys
        def get_key(item):
            if item['type'] == 'construction':
                return ('construction', item['uid'], item['u_type'], item['pos'])
            else:
                # 'production'
                return ('production', item["producer_type"],item['producer_id'], item['pos'], item['product'])
        
        gt_keys = [get_key(x) for x in gt_queue]
        pred_keys = [get_key(x) for x in pred_queue]
        
        # Multiset matching
        # Simple approach: Count occurrences of each key
        from collections import Counter
        gt_counts = Counter(gt_keys)
        pred_counts = Counter(pred_keys)
        
        # Intersection counts = Matches (TP)
        tp = 0
        all_keys = set(gt_keys) | set(pred_keys)
        
        # For Progress MAE, we need to map actual items.
        # Since we might have duplicates, we pair them up one-to-one greedily.
        
        progress_diffs = []
        
        # Group items by key for mapping
        gt_by_key = defaultdict(list)
        for x in gt_queue: gt_by_key[get_key(x)].append(x)
            
        pred_by_key = defaultdict(list)
        for x in pred_queue: pred_by_key[get_key(x)].append(x)
        
        for k in all_keys:
            g_list = gt_by_key[k]
            p_list = pred_by_key[k]
            
            n_g = len(g_list)
            n_p = len(p_list)
            
            matched = min(n_g, n_p)
            tp += matched
            
            # Calculate progress error for matched items
            # Since items with same key are identical except progress, order doesn't strictly matter 
            # unless progress distinguishes them further. 
            # Sort by progress to minimize difference (optimal matching for scalar 1D)
            g_progs = sorted([x['progress'] for x in g_list])
            p_progs = sorted([x['progress'] for x in p_list])
            
            for i in range(matched):
                progress_diffs.append(abs(g_progs[i] - p_progs[i]))
                
        # Acc = TP / (TP + FP + FN) ? Or just Accuracy?
        # Standard definition for set matching is often Jaccard or F1.
        # User said "Exact Match Acc". Usually means strict 1/0 for whole sequence?
        # "Queue: (Exact Match Acc) -> 0.78". This looks like a ratio.
        # Let's use F1 or Accuracy = TP / Max(LenGT, LenPred)? Or TP / (TP + FP + FN)?
        # User: "Exact match acc... constructing see uid..., production see ...".
        # This sounds like element-wise accuracy.
        # Let's use TP / (TP + FP + FN) = Jaccard Index basically, or F1.
        # The user specifically mentioned "Acc" which usually means matches / total_items.
        # Let's use F1-score style logic or TP/Union.
        
        # Total GT items = len(gt_queue)
        # Total Pred items = len(pred_queue)
        # TP = tp
        # FP = len(pred_queue) - tp
        # FN = len(gt_queue) - tp
        
        # If both empty -> 1.0
        if not gt_queue and not pred_queue:
            metrics['Dev_Queue_F1'] = 1.0
        elif not gt_queue and pred_queue:
            metrics['Dev_Queue_F1'] = 0.0
        elif gt_queue and not pred_queue:
            metrics['Dev_Queue_F1'] = 0.0
        else:
            # Let's use F1 as a robust 'Accuracy' metric for retrieval
            metrics['Dev_Queue_F1'] = calc_f1(tp, len(pred_queue) - tp, len(gt_queue) - tp)
            
        if progress_diffs:
            metrics['Dev_Progress_MAE'] = np.mean(progress_diffs)
            
        return metrics
