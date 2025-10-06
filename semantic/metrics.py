import os
import json
import pandas as pd
from collections import defaultdict

def eval_metrics(csv_path="similarity_records.csv"):
    """
    Calculate semantic unlearning success rate using the formula from the paper:
    
    Success if: cos(o_a, y_a) / (cos(o_a, y_a) + cos(o_a, y_b)) > cos(o_b, y_a) / (cos(o_b, y_a) + cos(o_b, y_b))
    
    Where:
    - o_b: model output before unlearning
    - o_a: model output after unlearning  
    - y_b: definition before cutoff (original meaning)
    - y_a: definition after cutoff (new meaning)
    """
    
    # Load similarity records
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run calculate_similarity.py first.")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Calculate success rates
    results = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))
    
    for _, row in df.iterrows():
        model = row["model"]
        prompt_style = row["prompt_style"]
        
        # Extract similarities
        sim_before_1 = row["similarity_before_1"]  # cos(o_b, y_b) - before unlearn vs new meaning
        sim_before_2 = row["similarity_before_2"]  # cos(o_b, y_a) - before unlearn vs original meaning
        sim_after_1 = row["similarity_after_1"]    # cos(o_a, y_b) - after unlearn vs new meaning  
        sim_after_2 = row["similarity_after_2"]    # cos(o_a, y_a) - after unlearn vs original meaning
        
        # Calculate success using the formula
        # Left side: cos(o_a, y_a) / (cos(o_a, y_a) + cos(o_a, y_b))
        left_numerator = sim_after_2
        left_denominator = sim_after_1 + sim_after_2
        
        # Right side: cos(o_b, y_a) / (cos(o_b, y_a) + cos(o_b, y_b))
        right_numerator = sim_before_2
        right_denominator = sim_before_1 + sim_before_2
        
        # Avoid division by zero
        if left_denominator > 0 and right_denominator > 0:
            left_score = left_numerator / left_denominator
            right_score = right_numerator / right_denominator
            
            # Success if left > right
            results[model][prompt_style]["total"] += 1
            if left_score > right_score:
                results[model][prompt_style]["success"] += 1
    
    # Calculate success rates
    success_rates = {}
    for model_name, model_data in results.items():
        success_rates[model_name] = {}
        for prompt_style, stats in model_data.items():
            total = stats["total"]
            success = stats["success"]
            success_rate = success / total if total > 0 else 0
            success_rates[model_name][prompt_style] = {
                "success_rate": success_rate,
                "success_count": success,
                "total_count": total
            }
    
    return success_rates

def print_results(success_rates):
    """Print results in a formatted table"""
    print("\n" + "="*80)
    print("SEMANTIC UNLEARN SUCCESS RATES")
    print("="*80)
    print(f"{'Model':<40} {'Prompt':<10} {'Success Rate':<15} {'Success/Total':<15}")
    print("-"*80)
    
    for model_name, model_data in success_rates.items():
        for prompt_style, stats in model_data.items():
            success_rate = stats["success_rate"]
            success_count = stats["success_count"]
            total_count = stats["total_count"]
            
            print(f"{model_name:<40} {prompt_style:<10} {success_rate:<15.3f} {success_count}/{total_count:<15}")
    
    print("="*80)

def save_results(success_rates, output_file="semantic_unlearn_success_rates.json"):
    """Save results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(success_rates, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    success_rates = eval_metrics()
    
    if success_rates:
        print_results(success_rates)
    else:
        print("Failed to calculate success rates. Please check the input data.")
