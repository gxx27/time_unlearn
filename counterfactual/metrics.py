import os
import json
from collections import defaultdict

def eval_metrics(output_path="results"):
    results = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))
    
    filename = "counterfactual.json"
    file_path = os.path.join(output_path, filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each model
        for model_name, model_data in data.items():
            # Process each prompt style
            for prompt_style, prompt_results in model_data.items():
                for result in prompt_results:
                    extracted_answer = result.get("extracted_answer", "N/A")
                    answer_before_unlearn = result.get("answer_before_unlearn", "N/A")
                    
                    # Count total attempts
                    results[model_name][prompt_style]["total"] += 1
                    
                    # Count successes (when extracted answer differs from before unlearn)
                    if extracted_answer != answer_before_unlearn:
                        results[model_name][prompt_style]["success"] += 1
                        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
    
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
    print("UNLEARN SUCCESS RATES")
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


if __name__ == "__main__":
    success_rates = eval_metrics()
    print_results(success_rates)
