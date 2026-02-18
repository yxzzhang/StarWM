import json
import re
import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluator import SC2WorldEvaluator
from metrics import calc_f1, calc_precision, calc_recall, solve_awd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_.*?\|>", "", text)
    text = text.replace("/no_think", "")
    return text.strip()

def extract_fields(data):
    """Helper to extract prompt, prediction, and ground truth from various JSON formats."""
    prompt = data.get('prompt') or data.get('input')
    pred = data.get('predict') or data.get('output')
    gt = data.get('label') or data.get('labels')
                
    return prompt, pred, gt

def main():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_file", type=str, required=True, help="Path to World Model jsonl")
    parser.add_argument("--zeroshot_32b_file", type=str, required=True, help="Path to Zero-shot 32B jsonl")
    parser.add_argument("--zeroshot_8b_file", type=str, default=None, help="Path to Zero-shot 8B jsonl")
    parser.add_argument("--static_bias_file", type=str, default=None, help="Path to Static Bias jsonl")
    parser.add_argument("--output_dir", type=str, default="offline_evaluate/results", help="Directory to save results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--figure2_frames", type=int, nargs='+', help="List of frames for Figure 2")
    args = parser.parse_args()
    
    sns.set_theme(style="whitegrid")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = SC2WorldEvaluator()
    
    # Define models dict
    models = {
        "Our World Model": args.wm_file,
        "Qwen3-32B": args.zeroshot_32b_file
    }
    if args.zeroshot_8b_file:
        models["Qwen3-8B"] = args.zeroshot_8b_file
    if args.static_bias_file:
        models["Static Bias"] = args.static_bias_file
    
    results = {}
    
    for model_name, file_path in models.items():
        print(f"Evaluating {model_name} from {file_path}...")
        
        agg_metrics = defaultdict(list)
        micro_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        hallucination_count = 0
        total_samples = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        total_samples = len(lines)
        
        for i, line in tqdm(enumerate(lines), total=total_samples, desc=f"Evaluating {model_name}"):

            data = json.loads(line)
            if not data:
                continue

            prompt, pred, gt = extract_fields(data)
            
            if not prompt or not pred or not gt:
                continue
            
            pred_obs = clean_text(pred)
            gt_obs = clean_text(gt)
            
            frame_metrics = evaluator.evaluate_frame(gt_obs, pred_obs)
            
            if "error" in frame_metrics:
                hallucination_count += 1
                continue
                
            # Aggregate Scalar Metrics
            for k, v in frame_metrics.items():
                if isinstance(v, (int, float)):
                    agg_metrics[k].append(v)
                elif k.startswith("Micro_Counts_"):
                    # Aggregate counts
                    grp = k.replace("Micro_Counts_", "")
                    micro_counts[grp]['tp'] += v['tp']
                    micro_counts[grp]['fp'] += v['fp']
                    micro_counts[grp]['fn'] += v['fn']
            
            # Store Frame Index for plotting (sequential)
            agg_metrics['Frame_Index'].append(i)
            # Store GT snippet for robust alignment
            # Using prompt or gt text snippet
            gt_snippet = clean_text(gt) if gt else ""
            agg_metrics['GT_Snippet'].append(gt_snippet)
            
        # Calculate Micro Avgs (Over dataset)
        final_metrics = {}
        for grp, counts in micro_counts.items():
            final_metrics[f'Micro_F1_{grp}'] = calc_f1(counts['tp'], counts['fp'], counts['fn'])
            final_metrics[f'Micro_Prec_{grp}'] = calc_precision(counts['tp'], counts['fp'])
            final_metrics[f'Micro_Recall_{grp}'] = calc_recall(counts['tp'], counts['fn'])
            
        # Calculate Macro Avgs (Mean of Frame metrics)
        for k, vals in agg_metrics.items():
            if k.startswith("Frame_"):
                final_metrics[k.replace("Frame_", "")] = np.mean(vals)
            elif k not in ['Frame_Index', 'Time_Seconds', 'GT_Snippet']:
                final_metrics[k] = np.mean(vals)
                
        # Keep Time Series for Plotting
        final_metrics['Time_Series'] = {
            'Time': agg_metrics.get('Time_Seconds', []),
            'GT_Snippet': agg_metrics.get('GT_Snippet', []),
            'AWD_Self': [ (agg_metrics.get('Macro_AWD_Self_Unit', [0])[i] + agg_metrics.get('Macro_AWD_Self_Struct', [0])[i])/2 for i in range(len(agg_metrics.get('Frame_Index', []))) ],
            'AWD_Enemy': [ (agg_metrics.get('Macro_AWD_Enemy_Unit', [0])[i] + agg_metrics.get('Macro_AWD_Enemy_Struct', [0])[i] + agg_metrics.get('Macro_AWD_Snap_Enemy_Struct', [0])[i])/3 for i in range(len(agg_metrics.get('Frame_Index', []))) ]
        }
        
        results[model_name] = {
            "metrics": final_metrics,
            "hallucinations": hallucination_count,
            "total": total_samples
        }
        
    # --- Presentation ---
    print("\n" + "="*60)
    print("SC2 World Model Evaluation Report (V2)")
    print("="*60)
    
    # 1. Hallucination Report
    for m, res in results.items():
        h = res['hallucinations']
        t = res['total']
        print(f"[{m}] Hallucinations (Skipped): {h}/{t} ({h/t*100:.2f}%)")
    
    print("-" * 60)
    
    # 2. Main Table
    print(f"{'Method':<20} | {'Macro AWD (Self/Enemy)':<25} | {'Micro F1 (Self/Enemy)':<25} | {'Micro-Entity (HP)':<20} | {'Econ SMAPE (Min/Gas)':<20} | {'Dev (Queue F1/MAE)':<20}")
    print("-" * 140)
    
    # Custom Order
    display_order = ["Static Bias", "Qwen3-8B", "Qwen3-32B", "Our World Model"]
    
    for m_name in display_order:
        if m_name not in results:
            continue
            
        res = results[m_name]
        m = res['metrics']
        
        awd_self = (m.get('Macro_AWD_Self_Unit', 0) + m.get('Macro_AWD_Self_Struct', 0)) / 2
        awd_enemy = (m.get('Macro_AWD_Enemy_Unit', 0) + m.get('Macro_AWD_Enemy_Struct', 0) + m.get('Macro_AWD_Snap_Enemy_Struct', 0)) / 3
        
        # Micro F1
        f1_self = (m.get('Micro_F1_Self_Unit', 0) + m.get('Micro_F1_Self_Struct', 0)) / 2
        f1_enemy = (m.get('Micro_F1_Enemy_Unit', 0) + m.get('Micro_F1_Enemy_Struct', 0) + m.get('Micro_F1_Snap_Enemy_Struct', 0)) / 3

        # Micro Entity (HP MAE)
        # Note: HP MAE is defined as Frame_{name}_HP_MAE in evaluator, aggregated to {name}_HP_MAE in run_eval
        # We want Self / Enemy
        hp_self = m.get('Self_Unit_HP_MAE', 0) # Only Unit HP usually relevant? Or struct too?
        # User said "Micro-Entity (self/enemy unit HP)" -> implies Unit HP only.
        hp_enemy = m.get('Enemy_Unit_HP_MAE', 0)
        
        econ_min = m.get('Resource_Minerals_SMAPE', 0)
        econ_gas = m.get('Resource_Gas_SMAPE', 0)
        
        dev_acc = m.get('Dev_Queue_F1', 0)
        dev_mae = m.get('Dev_Progress_MAE', 0)
        
        print(f"{m_name:<20} | {awd_self:.2f} / {awd_enemy:.2f}{' '*11} | {f1_self:.2f} / {f1_enemy:.2f}{' '*11} | {hp_self:.2f} / {hp_enemy:.2f}{' '*6} | {econ_min:.2f} / {econ_gas:.2f}{' '*6} | {dev_acc:.2f} / {dev_mae:.2f}")
    
    print("\n\n\n")
    # 3. Generate Transposed Text Table
    report_lines = []
    
    metric_map = [
        ("Macro-Dynamics Consistency", None),
        ("Macro_AWD_Self_Unit", "Self Unit AWD"),
        ("Macro_AWD_Self_Struct", "Self Struct AWD"),
        ("Macro_AWD_Enemy_Unit", "Enemy Unit AWD"),
        ("Macro_AWD_Enemy_Struct", "Enemy Struct AWD"),
        ("Macro_AWD_Snap_Enemy_Struct", "Snap Enemy Struct AWD"),
        
        ("Global Economy & Status", None),
        ("Resource_Minerals_SMAPE", "Minerals SMAPE"),
        ("Resource_Minerals_Rate_SMAPE", "Minerals Rate SMAPE"),
        ("Resource_Gas_SMAPE", "Gas SMAPE"),
        ("Resource_Gas_Rate_SMAPE", "Gas Rate SMAPE"),
        ("Supply_Used_SMAPE", "Supply Used SMAPE"),
        ("Supply_Cap_SMAPE", "Supply Cap SMAPE"),
        ("Status_Alerts_F1", "Alerts F1"),
        ("Tech_Upgrades_F1", "Upgrades F1"),
        ("Econ_Workers_SMAPE", "Workers num SMAPE"),
        
        ("Development", None),
        ("Dev_Queue_F1", "Queue F1"),
        ("Dev_Progress_MAE", "Progress MAE"),
        
        ("Micro-Unit Attributes", None),
        # Remove "Micro" from keys as per instruction
        ("Micro_F1_Self_Unit", "Self Unit F1 (Micro_avg)"),
        ("Micro_Prec_Self_Unit", "Self Unit Prec (Micro_avg)"),
        ("Micro_Recall_Self_Unit", "Self Unit Recall (Micro_avg)"),
        ("F1_Self_Unit", "Self Unit F1"),
        ("Prec_Self_Unit", "Self Unit Prec"),
        ("Recall_Self_Unit", "Self Unit Recall"),
        ("Micro_F1_Self_Struct", "Self Struct F1 (Micro_avg)"),
        ("Micro_Prec_Self_Struct", "Self Struct Prec (Micro_avg)"),
        ("Micro_Recall_Self_Struct", "Self Struct Recall (Micro_avg)"),
        ("F1_Self_Struct", "Self Struct F1"),
        ("Prec_Self_Struct", "Self Struct Prec"),
        ("Recall_Self_Struct", "Self Struct Recall"),
        ("Micro_F1_Enemy_Unit", "Enemy Unit F1 (Micro_avg)"),
        ("Micro_Prec_Enemy_Unit", "Enemy Unit Prec (Micro_avg)"),
        ("Micro_Recall_Enemy_Unit", "Enemy Unit Recall (Micro_avg)"),
        ("F1_Enemy_Unit", "Enemy Unit F1"),
        ("Prec_Enemy_Unit", "Enemy Unit Prec"),
        ("Recall_Enemy_Unit", "Enemy Unit Recall"),
        ("Micro_F1_Enemy_Struct", "Enemy Struct F1 (Micro_avg)"),
        ("Micro_Prec_Enemy_Struct", "Enemy Struct Prec (Micro_avg)"),
        ("Micro_Recall_Enemy_Struct", "Enemy Struct Recall (Micro_avg)"),
        ("F1_Enemy_Struct", "Enemy Struct F1"),
        ("Prec_Enemy_Struct", "Enemy Struct Prec"),
        ("Recall_Enemy_Struct", "Enemy Struct Recall"),
        ("Micro_F1_Snap_Enemy_Struct", "Snap Enemy Struct F1 (Micro_avg)"),
        ("Micro_Prec_Snap_Enemy_Struct", "Snap Enemy Struct Prec (Micro_avg)"),
        ("Micro_Recall_Snap_Enemy_Struct", "Snap Enemy Struct Recall (Micro_avg)"),
        ("F1_Snap_Enemy_Struct", "Snap Enemy Struct F1"),
        ("Prec_Snap_Enemy_Struct", "Snap Enemy Struct Prec"),
        ("Recall_Snap_Enemy_Struct", "Snap Enemy Struct Recall"),

        ("Self_Unit_HP_MAE", "Self Unit HP MAE"),
        ("Self_Unit_Energy_MAE", "Self Unit Energy MAE"),
        ("Self_Struct_HP_MAE", "Self Struct HP MAE"),
        ("Self_Struct_Energy_MAE", "Self Struct Energy MAE"),
        ("Enemy_Unit_HP_MAE", "Enemy Unit HP MAE"),
        ("Enemy_Unit_Energy_MAE", "Enemy Unit Energy MAE"),
        ("Enemy_Struct_HP_MAE", "Enemy Struct HP MAE"),
        # ("Enemy_Struct_Energy_MAE", "Enemy Struct Energy MAE"),
    ]
    
    model_names = list(models.keys()) # ["SFT World Model", "Zero-shot 32B", "Zero-shot 8B", "Static Bias"]
    
    # Header
    header = f"{'Metric':<40}"
    for m in model_names:
        header += f" | {m:<20}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    for key, display_name in metric_map:
        if display_name is None:
            # Section Header
            report_lines.append(f"\n[ {key} ]")
            continue
            
        row = f"{display_name:<40}"
        for m in model_names:
            val = results[m]['metrics'].get(key, None)
            if val is not None:
                row += f" | {val:.2f}{' '*16}"
            else:
                row += f" | {'N/A':<20}"
        report_lines.append(row)
        
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(os.path.join(args.output_dir, "evaluation_report.txt"), 'w') as f:
        f.write(report_text)
        
    # 2. Figure 1: Dynamics Consistency (2 Subplots)
    # Split: SFT vs ZS 32B, and SFT vs ZS 8B
    
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        def plot_time_series(ax, m1_name, m2_name):
            if m1_name not in results or m2_name not in results:
                return
                
            # Helper to get series
            def get_series(m_name):
                ts = results[m_name]['metrics']['Time_Series']
                times = np.array(ts['Time'])
                awd_self = np.array(ts['AWD_Self'])
                awd_enemy = np.array(ts['AWD_Enemy'])
                
                if len(times) == 0: return None, None, None
                
                sort_idx = np.argsort(times)
                return times[sort_idx], awd_self[sort_idx], awd_enemy[sort_idx]
                
            t1, s1_self, s1_enemy = get_series(m1_name)
            t2, s2_self, s2_enemy = get_series(m2_name)
            
            if t1 is None or t2 is None: return

            # Smoothing
            window = 30
            def moving_average(x, w):
                if len(x) < w: return x
                # Use valid to avoid boundary effects, but we need to trim time too
                return np.convolve(x, np.ones(w), 'valid') / w
            
            # We need common time basis for fill_between if we want it precise,
            # but here we can just smooth and plot independently if they are roughly aligned.
            # Assuming they are from same dataset, time points should be identical.
            # Let's truncate to min length after smoothing
            
            s1_total = s1_self + s1_enemy # Total AWD for simplified comparison curve
            s2_total = s2_self + s2_enemy
            
            s1_smooth = moving_average(s1_total, window)
            s2_smooth = moving_average(s2_total, window)
            t_smooth = t1[window-1:]
            
            # Make sure lengths match
            min_len = min(len(s1_smooth), len(s2_smooth), len(t_smooth))
            s1_smooth = s1_smooth[:min_len]
            s2_smooth = s2_smooth[:min_len]
            t_smooth = t_smooth[:min_len]
            
            # Plot SFT (m1)
            ax.plot(t_smooth, s1_smooth, label=f"{m1_name}", color='blue', linewidth=2.5)
            
            # Plot ZS (m2)
            ax.plot(t_smooth, s2_smooth, label=f"{m2_name}", color='red', linewidth=2, linestyle='--')
            
            # Fill Between
            # Green where SFT < ZS (Gain)
            ax.fill_between(t_smooth, s1_smooth, s2_smooth, where=(s2_smooth > s1_smooth), 
                            interpolate=True, color='green', alpha=0.15, label='SFT Superiority')
            
            # Red where SFT > ZS (Loss - shouldn't happen much)
            ax.fill_between(t_smooth, s1_smooth, s2_smooth, where=(s2_smooth <= s1_smooth), 
                            interpolate=True, color='red', alpha=0.1, label='SFT Inferiority')
                
            ax.set_title(f"Dynamics Consistency: {m1_name} vs {m2_name}", fontsize=18)
            ax.set_xlabel("Game Time (Seconds)", fontsize=16)
            ax.set_ylabel("Augmented Wasserstein Distance (AWD)", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            
            if len(t_smooth) > 0:
                start_t = int(min(t_smooth))
                end_t = int(max(t_smooth))
                step = max(60, (end_t - start_t) // 8)
                ax.set_xticks(np.arange(start_t, end_t + step, step))

        plot_time_series(ax1, "Our World Model", "Qwen3-32B")
        if "Qwen3-8B" in results:
            plot_time_series(ax2, "Our World Model", "Qwen3-8B")
        else:
            ax2.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "figure1_dynamics_consistency.pdf"), format='pdf', dpi=300, bbox_inches='tight')
        print(f"Figure 1 saved to {args.output_dir}/figure1_dynamics_consistency.pdf")
    
    # 2.5 Figure 1-1: Dynamics Consistency Split (Self vs Enemy)
    # Only SFT vs Zero-shot 32B
    
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        m1_name = "Our World Model"
        m2_name = "Qwen3-32B"
        
        def plot_split_metric(ax, metric_key, title_suffix):
            if m1_name not in results or m2_name not in results:
                return

            def get_single_series(m_name, key):
                ts = results[m_name]['metrics']['Time_Series']
                times = np.array(ts['Time'])
                vals = np.array(ts[key])
                if len(times) == 0: return None, None
                sort_idx = np.argsort(times)
                return times[sort_idx], vals[sort_idx]

            t1, v1 = get_single_series(m1_name, metric_key)
            t2, v2 = get_single_series(m2_name, metric_key)

            if t1 is None or t2 is None: return

            # Smoothing
            window = 30
            def moving_average(x, w):
                if len(x) < w: return x
                return np.convolve(x, np.ones(w), 'valid') / w

            v1_smooth = moving_average(v1, window)
            v2_smooth = moving_average(v2, window)
            t_smooth = t1[window-1:]

            min_len = min(len(v1_smooth), len(v2_smooth), len(t_smooth))
            v1_smooth = v1_smooth[:min_len]
            v2_smooth = v2_smooth[:min_len]
            t_smooth = t_smooth[:min_len]

            # Plot SFT
            ax.plot(t_smooth, v1_smooth, label=f"Our World Model", color='blue', linewidth=2.5)
            # Plot ZS
            ax.plot(t_smooth, v2_smooth, label=f"Qwen3-32B", color='red', linewidth=2, linestyle='--')

            # Fill
            ax.fill_between(t_smooth, v1_smooth, v2_smooth, where=(v2_smooth > v1_smooth),
                            interpolate=True, color='green', alpha=0.15, label='WM Superiority')
            ax.fill_between(t_smooth, v1_smooth, v2_smooth, where=(v2_smooth <= v1_smooth),
                            interpolate=True, color='red', alpha=0.1, label='WM Inferiority')

            ax.set_title(f"Macro-Situation Consistency ({title_suffix})", fontsize=18)
            ax.set_xlabel("Game Time (Seconds)", fontsize=16)
            ax.set_ylabel(f"Augmented Wasserstein Distance (AWD)", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            
            if len(t_smooth) > 0:
                start_t = int(min(t_smooth))
                end_t = int(max(t_smooth))
                step = max(60, (end_t - start_t) // 8)
                ax.set_xticks(np.arange(start_t, end_t + step, step))

        plot_split_metric(ax1, 'AWD_Self', "Self")
        plot_split_metric(ax2, 'AWD_Enemy', "Enemy")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "figure1_1_dynamics_split.pdf"), format='pdf', dpi=300, bbox_inches='tight')
        print(f"Figure 1-1 saved to {args.output_dir}/figure1_1_dynamics_split.pdf")
    
    # 3. Figure 2: Qualitative Visualization (Case Study)
    # Heuristic: Max diff between SFT and ZS 32B
    if args.plot:
        sft_data = results.get("Our World Model", {}).get('metrics', {}).get('Time_Series', {})
        zs_data = results.get("Qwen3-32B", {}).get('metrics', {}).get('Time_Series', {})
        
        if sft_data and zs_data:
            # Align data based on Time and GT_Snippet (to handle shuffled ZS file)
            # SFT is reference (ordered)
            
            times_sft = np.array(sft_data['Time'])
            
            # Build ZS Lookup: (Time, Snippet) -> Total AWD
            zs_lookup = {}
            zs_raw_awd = np.array(zs_data['AWD_Self']) + np.array(zs_data['AWD_Enemy'])
            for t, snip, val in zip(zs_data['Time'], zs_data['GT_Snippet'], zs_raw_awd):
                zs_lookup[(t, snip)] = val
                
            # Construct aligned arrays
            aligned_zs_awd = []
            valid_indices = []
            
            for i, (t, snip) in enumerate(zip(sft_data['Time'], sft_data['GT_Snippet'])):
                if (t, snip) in zs_lookup:
                    aligned_zs_awd.append(zs_lookup[(t, snip)])
                    valid_indices.append(i)
                else:
                    # Mismatch or missing frame
                    continue
            assert len(sft_data['Time']) == len(zs_data['Time']) == len(valid_indices)

            valid_indices = np.array(valid_indices)
            
            target_frames = []

            if args.figure2_frames:
                 target_frames = args.figure2_frames
            elif len(valid_indices) > 0:
                aligned_zs_awd = np.array(aligned_zs_awd)
                # SFT AWD at valid indices
                sft_raw_awd = np.array(sft_data['AWD_Self']) + np.array(sft_data['AWD_Enemy'])
                aligned_sft_awd = sft_raw_awd[valid_indices]
                
                # Calculate Diff on ALIGNED data
                diff = aligned_zs_awd - aligned_sft_awd
                
                # Refined Heuristic:
                # 1. Late game (Time > 300)
                valid_times = times_sft[valid_indices]
                late_mask = (valid_times > 300)
                
                best_idx = -1
                
                # Filter valid indices by late_mask
                # Note: diff corresponds to valid_indices
                
                target_indices = []
                
                if np.any(late_mask):
                    # Indices in 'diff' array where time > 300
                    late_indices_in_diff = np.where(late_mask)[0]
                    
                    # Sort these by diff descending
                    sorted_late_sub_indices = late_indices_in_diff[np.argsort(diff[late_indices_in_diff])[::-1]]
                    
                    # Map back to original SFT indices
                    target_indices = valid_indices[sorted_late_sub_indices]
                else:
                    # Fallback to all valid
                    sorted_sub_indices = np.argsort(diff)[::-1]
                    target_indices = valid_indices[sorted_sub_indices]
                
                for idx in target_indices:
                    if len(target_frames) >= 5:
                        break
                    if idx not in target_frames:
                        target_frames.append(idx)

            if target_frames:
                print(f"Generating Figure 2 for frames: {target_frames}")

                # Load ZS file into a Map for random access by Prompt/Input
                print("Loading Zero-shot 32B file for alignment...")
                zs_file_map = {}
                with open(args.zeroshot_32b_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            k, _, _ = extract_fields(d)
                            if k: zs_file_map[clean_text(k)] = d 
                        except:
                            pass

                zs_8b_file_map = {}
                if args.zeroshot_8b_file:
                    print("Loading Zero-shot 8B file for alignment...")
                    with open(args.zeroshot_8b_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                d = json.loads(line)
                                k, _, _ = extract_fields(d)
                                if k: zs_8b_file_map[clean_text(k)] = d
                            except:
                                pass

                def get_wm_line(idx):
                    # SFT file is ordered, so we can use line index
                    with open(args.wm_file, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i == idx: return json.loads(line)
                    return None

                for frame_idx in target_frames:
                    print(f"Plotting Frame {frame_idx} (Time {times_sft[frame_idx]}s)...")
                    
                    wm_line = get_wm_line(frame_idx)
                    if not wm_line: continue
                    
                    # Find corresponding ZS line
                    wm_input = wm_line.get('prompt') or wm_line.get('input')
                    clean_input = clean_text(wm_input)
                    zs_line = zs_file_map.get(clean_input)
                    
                    zs_8b_line = None
                    if args.zeroshot_8b_file:
                        zs_8b_line = zs_8b_file_map.get(clean_input)
                    
                    if not zs_line:
                        print(f"Warning: Could not find matching ZS line for Frame {frame_idx}")
                        continue
                        
                    # Verify Labels Match (Double Check)
                    wm_label = clean_text(wm_line.get('label') or wm_line.get('labels'))
                    zs_label = clean_text(zs_line.get('label') or zs_line.get('labels'))
                    
                    # Use a soft check or assert
                    if wm_label != zs_label:
                        # Sometimes minor whitespace diffs?
                        # if wm_label[:50] != zs_label[:50]:
                        #      print(f"Warning: Label mismatch for Frame {frame_idx}!")
                             # continue # Optional: skip if strict
                        print(f"Warning: Label mismatch (32B) for Frame {frame_idx}!")
                        raise ValueError(f"WM Label: {wm_label}\nZS Label: {zs_label}")
                        continue
                        # raise ValueError(f"WM Label: {wm_label}\nZS Label: {zs_label}")
                    
                    if zs_8b_line:
                        zs_8b_label = clean_text(zs_8b_line.get('label') or zs_8b_line.get('labels'))
                        if wm_label != zs_8b_label:
                             print(f"Warning: Label mismatch (8B) for Frame {frame_idx}!")
                             raise ValueError(f"WM Label: {wm_label}\nZS Label: {zs_8b_label}")
                             continue

                    if wm_line:
                        from parser import SC2TextParser
                        p = SC2TextParser()
                        
                        gt_text = wm_label
                        sft_text = clean_text(wm_line.get('predict') or wm_line.get('output'))
                        zs_text = clean_text(zs_line.get('predict') or zs_line.get('output')) if zs_line else ""
                        zs_8b_text = clean_text(zs_8b_line.get('predict') or zs_8b_line.get('output')) if zs_8b_line else ""
                        
                        gt_data = p.parse(gt_text)
                        sft_data = p.parse(sft_text)
                        zs_data = p.parse(zs_text) if zs_text else None
                        zs_8b_data = p.parse(zs_8b_text) if zs_8b_text else None
                        
                        if gt_data and sft_data and zs_data:
                            if zs_8b_data:
                                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
                            else:
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                            
                            def normalize_pos(pos):
                                # Playable area [12, 10] to [76, 74] -> 64x64
                                x, y = pos
                                nx = max(0, min(64, x - 12))
                                ny = max(0, min(64, y - 10))
                                return (nx, ny)

                            def plot_units(ax, u_list, color, marker, label, filled=True, alpha=0.7, size=80):
                                if not u_list: return
                                raw_pos = [u.pos for u in u_list]
                                if not raw_pos: return
                                
                                norm_pos = np.array([normalize_pos(p) for p in raw_pos])
                                
                                # Style Definition
                                if filled:
                                    # Ground Truth: Lighter, Filled, No Edge
                                    ax.scatter(norm_pos[:,0], norm_pos[:,1], 
                                             c=color, marker=marker, s=size, label=label, 
                                             alpha=0.4, edgecolors='none') # Increased alpha for visibility
                                else:
                                    # Prediction: Darker/Vivid, Hollow, Thick Edge
                                    ax.scatter(norm_pos[:,0], norm_pos[:,1], 
                                             facecolors='none', edgecolors=color, marker=marker, s=size*1.2, label=label, 
                                             alpha=1.0, linewidths=2)

                            def draw_scene(ax, g_d, p_d, title, show_legend=False):
                                ax.set_xlim(0, 64)
                                ax.set_ylim(0, 64)
                                ax.set_aspect('equal')
                                
                                # Scheme:
                                # Self = Blue, Enemy = Red
                                # GT = Solid Shapes (Circle=Unit, Square=Struct)
                                # Pred = Hollow Shapes (Circle=Unit, Square=Struct)
                                
                                # GT Self Unit (Blue Circle Solid)
                                plot_units(ax, [u for u in g_d['units']], 'blue', 'o', 'GT Self Unit', filled=True)
                                # GT Self Struct (Blue Square Solid)
                                plot_units(ax, [u for u in g_d['structures']], 'blue', 's', 'GT Self Struct', filled=True)
                                
                                # GT Enemy Unit (Red Circle Solid)
                                plot_units(ax, [u for u in g_d['enemies'] if u.category=='EnemyUnit'], 'red', 'o', 'GT Enemy Unit', filled=True)
                                # GT Enemy Struct (Red Square Solid)
                                plot_units(ax, [u for u in g_d['enemies'] if u.category=='EnemyStruct'], 'red', 's', 'GT Enemy Struct', filled=True)
                                # GT Snap Enemy Struct (DarkOrange Square Solid)
                                plot_units(ax, [u for u in g_d['enemies'] if u.category=='SnapEnemyStruct'], 'darkorange', 's', 'GT Snap Struct', filled=True)
                                
                                # Pred Self Unit (Blue Circle Hollow)
                                plot_units(ax, [u for u in p_d['units']], 'blue', 'o', 'Pred Self Unit', filled=False)
                                # Pred Self Struct (Blue Square Hollow)
                                plot_units(ax, [u for u in p_d['structures']], 'blue', 's', 'Pred Self Struct', filled=False)
                                
                                # Pred Enemy Unit (Red Circle Hollow)
                                plot_units(ax, [u for u in p_d['enemies'] if u.category=='EnemyUnit'], 'red', 'o', 'Pred Enemy Unit', filled=False)
                                # Pred Enemy Struct (Red Square Hollow)
                                plot_units(ax, [u for u in p_d['enemies'] if u.category=='EnemyStruct'], 'red', 's', 'Pred Enemy Struct', filled=False)
                                # Pred Snap Enemy Struct (DarkOrange Square Hollow)
                                plot_units(ax, [u for u in p_d['enemies'] if u.category=='SnapEnemyStruct'], 'darkorange', 's', 'Pred Snap Struct', filled=False)

                                ax.set_title(title, fontsize=18)
                                ax.tick_params(axis='both', which='major', labelsize=14)
                                ax.grid(True, alpha=0.2)
                                
                                if show_legend:
                                    from matplotlib.lines import Line2D
                                    legend_elements = [
                                        # GT (Filled, lighter, no edge)
                                        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='GT Self Unit', markersize=10, alpha=0.4, linestyle='None', markeredgewidth=0),
                                        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', label='GT Self Struct', markersize=10, alpha=0.4, linestyle='None', markeredgewidth=0),
                                        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='GT Enemy Unit', markersize=10, alpha=0.4, linestyle='None', markeredgewidth=0),
                                        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='GT Enemy Struct', markersize=10, alpha=0.4, linestyle='None', markeredgewidth=0),
                                        Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', label='GT Snap Struct', markersize=10, alpha=0.4, linestyle='None', markeredgewidth=0),
                                        
                                        # Pred (Hollow, vivid, thick edge)
                                        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='blue', label='Pred Self Unit', markersize=12, markeredgewidth=2, alpha=1.0, linestyle='None'),
                                        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='blue', label='Pred Self Struct', markersize=12, markeredgewidth=2, alpha=1.0, linestyle='None'),
                                        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red', label='Pred Enemy Unit', markersize=12, markeredgewidth=2, alpha=1.0, linestyle='None'),
                                        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='red', label='Pred Enemy Struct', markersize=12, markeredgewidth=2, alpha=1.0, linestyle='None'),
                                        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='darkorange', label='Pred Snap Struct', markersize=12, markeredgewidth=2, alpha=1.0, linestyle='None'),
                                    ]
                                    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.7)

                            if zs_8b_data:
                                draw_scene(ax1, gt_data, zs_8b_data, f"Qwen3-8B Prediction (T={times_sft[frame_idx]}s)", show_legend=True)
                                draw_scene(ax2, gt_data, zs_data, f"Qwen3-32B Prediction (T={times_sft[frame_idx]}s)", show_legend=False)
                                draw_scene(ax3, gt_data, sft_data, f"Our World Model Prediction (T={times_sft[frame_idx]}s)", show_legend=False)
                            else:
                                draw_scene(ax1, gt_data, zs_data, f"Qwen3-32B Prediction (T={times_sft[frame_idx]}s)", show_legend=True)
                                draw_scene(ax2, gt_data, sft_data, f"Our World Model Prediction (T={times_sft[frame_idx]}s)", show_legend=False)
                            
                            plt.tight_layout()
                            fname = f"figure2_case_study_frame_{frame_idx}.pdf"
                            plt.savefig(os.path.join(args.output_dir, fname), format='pdf', dpi=300, bbox_inches='tight')
                            print(f"Saved {fname}")
                            plt.close()

    # Save JSON Summary
    clean_report = {}
    for m_name, res in results.items():
        clean_report[m_name] = {k: v for k, v in res['metrics'].items() if isinstance(v, (int, float))}
        
    with open(os.path.join(args.output_dir, "metrics_summary.json"), 'w') as f:
        json.dump(clean_report, f, indent=4)

if __name__ == "__main__":
    main()
