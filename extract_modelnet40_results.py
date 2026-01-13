#!/usr/bin/env python3
"""
Extract ModelNet40 experiment results from TensorBoard logs
"""

import os
import json
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_metrics(log_dir):
    """Extract metrics from TensorBoard logs"""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Directory not found: {log_dir}")
        return None
        
    # Find TensorBoard event files
    event_files = list(log_path.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None
        
    # Use the most recent event file
    latest_event = max(event_files, key=lambda x: x.stat().st_mtime)
    print(f"Reading from: {latest_event}")
    
    try:
        # Load TensorBoard data
        ea = EventAccumulator(str(latest_event))
        ea.Reload()
        
        metrics = {}
        
        # Extract scalar metrics
        scalar_tags = ea.Tags()['scalars']
        print(f"Available scalar tags: {scalar_tags}")
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            if scalar_events:
                # Get all values and steps
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                
                # Store final value, max value, and full history
                metrics[f"{tag}_final"] = values[-1] if values else None
                metrics[f"{tag}_max"] = max(values) if values else None
                metrics[f"{tag}_min"] = min(values) if values else None
                metrics[f"{tag}_values"] = values
                metrics[f"{tag}_steps"] = steps
                
        return metrics
        
    except Exception as e:
        print(f"Error reading TensorBoard data: {e}")
        return None

def extract_config_info(config_file):
    """Extract configuration from config.txt"""
    if not Path(config_file).exists():
        return {}
        
    config_info = {}
    try:
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Extract key metrics
        import re
        patterns = {
            'input_dim': r'Input dimension:\s*(\d+)',
            'total_parameters': r'Total trainable parameters:\s*([\d,]+)',
            'training_start': r'Training start time:\s*([^\n]+)',
            'epochs': r'Number of epochs:\s*(\d+)',
            'batch_size': r'Batch size:\s*(\d+)',
            'learning_rate': r'Learning rate:\s*([\d.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                if key == 'total_parameters':
                    config_info[key] = int(match.group(1).replace(',', ''))
                elif key in ['input_dim', 'epochs', 'batch_size']:
                    config_info[key] = int(match.group(1))
                elif key == 'learning_rate':
                    config_info[key] = float(match.group(1))
                else:
                    config_info[key] = match.group(1).strip()
                    
    except Exception as e:
        print(f"Error reading config: {e}")
        
    return config_info

def analyze_modelnet40_experiments():
    """Analyze all ModelNet40 experiments"""
    logs_dir = Path("logs")
    
    # Look for ModelNet40 experiment directories
    modelnet_dirs = [d for d in logs_dir.iterdir() 
                    if d.is_dir() and 'modelnet40' in d.name.lower()]
    
    print(f"Found ModelNet40 experiments: {[d.name for d in modelnet_dirs]}")
    
    results = []
    
    for exp_dir in modelnet_dirs:
        print(f"\n=== Analyzing {exp_dir.name} ===")
        
        # Extract TensorBoard metrics
        tb_metrics = extract_tensorboard_metrics(exp_dir)
        
        # Extract config info
        config_file = exp_dir / "config.txt"
        config_info = extract_config_info(config_file)
        
        # Compile results
        result = {
            'experiment': exp_dir.name,
            'directory': str(exp_dir)
        }
        
        # Add config info
        result.update(config_info)
        
        # Add TensorBoard metrics
        if tb_metrics:
            # Look for accuracy and loss metrics
            for key, value in tb_metrics.items():
                if 'final' in key or 'max' in key or 'min' in key:
                    result[key] = value
                    
            # Specifically look for top-1 accuracy
            acc_keys = [k for k in tb_metrics.keys() 
                       if 'acc' in k.lower() and ('val' in k.lower() or 'test' in k.lower())]
            
            if acc_keys:
                print(f"Found accuracy metrics: {acc_keys}")
                
                # Find the best accuracy
                best_acc = 0
                best_acc_key = None
                for key in acc_keys:
                    if 'max' in key and tb_metrics[key] > best_acc:
                        best_acc = tb_metrics[key]
                        best_acc_key = key
                    elif 'final' in key and best_acc_key is None:
                        best_acc = tb_metrics[key]
                        best_acc_key = key
                        
                result['best_accuracy'] = best_acc
                result['best_accuracy_source'] = best_acc_key
                print(f"Best accuracy: {best_acc:.4f} ({best_acc_key})")
            
        else:
            print("No TensorBoard metrics found")
            
        results.append(result)
    
    return results

def main():
    print("🔍 Extracting ModelNet40 Results...")
    print("=" * 50)
    
    results = analyze_modelnet40_experiments()
    
    if not results:
        print("❌ No ModelNet40 results found")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(results)
    
    # Display summary
    print("\n📊 MODELNET40 RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by best accuracy if available
    if 'best_accuracy' in df.columns:
        df_sorted = df.sort_values('best_accuracy', ascending=False)
        
        print("\n🏆 TOP ACCURACIES:")
        for idx, row in df_sorted.iterrows():
            if pd.notna(row.get('best_accuracy')):
                acc_pct = row['best_accuracy'] * 100
                print(f"  {row['experiment']}: {acc_pct:.2f}%")
        
        # Find overall best
        best_exp = df_sorted.iloc[0]
        best_acc_pct = best_exp['best_accuracy'] * 100
        print(f"\n🥇 BEST OVERALL ACCURACY: {best_acc_pct:.2f}% ({best_exp['experiment']})")
    
    # Display detailed table
    print(f"\n📋 DETAILED RESULTS:")
    display_cols = ['experiment', 'best_accuracy', 'total_parameters', 'epochs', 'batch_size', 'learning_rate']
    available_cols = [col for col in display_cols if col in df.columns]
    
    if available_cols:
        print(df[available_cols].to_string(index=False))
    else:
        print(df.to_string(index=False))
    
    # Save results
    output_file = "modelnet40_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    results = main()
