# %%
"""
Analyze blind comparison results and generate bar charts.
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = "data"

def load_comparison_results(data_dir=DATA_DIR):
    """Load all blind comparison result files."""
    data_path = Path(data_dir)
    result_files = list(data_path.glob("blind_comparison_results_*.json"))
    
    if not result_files:
        print(f"No comparison result files found in {data_dir}")
        print("Make sure you've run the blind comparison script first!")
        return []
    
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append({
                    'filename': file_path.name,
                    'data': data
                })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def analyze_single_session(data):
    """Analyze results from a single session."""
    summary = data.get('summary', {})
    
    return {
        'staircase_wins': summary.get('staircase_wins', 0),
        'topk_wins': summary.get('topk_wins', 0),
        'indifferent': summary.get('indifferent', 0),
        'total_comparisons': summary.get('total_comparisons', 0),
        'user_name': data.get('user_name', 'Unknown'),
        'session_start': data.get('session_start', 'Unknown')
    }

def create_single_session_chart(analysis, filename):
    """Create a bar chart for a single session."""
    categories = ['Staircase Wins', 'TopK Wins', 'Indifferent']
    values = [analysis['staircase_wins'], analysis['topk_wins'], analysis['indifferent']]
    colors = ['#2E8B57', '#CD5C5C', '#708090']  # Sea green, Indian red, Slate gray
    
    # Calculate error bars (Poisson for small counts, Binomial for larger samples)
    total = analysis['total_comparisons']
    error_bars = []
    
    for value in values:
        if total > 30 and value > 5:  # Use binomial approximation for larger samples
            # Binomial standard error: sqrt(n * p * (1-p))
            p = value / total if total > 0 else 0
            error = np.sqrt(total * p * (1 - p)) if total > 0 else 0
        else:  # Use Poisson approximation for small counts
            # Poisson standard error: sqrt(count)
            error = np.sqrt(value) if value > 0 else 0
        error_bars.append(error)
    
    # Print copyable values for this session
    print(f"\n📊 COPYABLE VALUES - {analysis['user_name']}")
    print(f"Total comparisons: {total}")
    for cat, val, err in zip(categories, values, error_bars):
        print(f"{cat}: {val} ± {err:.2f}")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', 
                   yerr=error_bars, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add value labels on bars
    for bar, value, error in zip(bars, values, error_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.2,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Blind Comparison Results - {analysis["user_name"]}\n'
              f'Total Comparisons: {analysis["total_comparisons"]} (with {("Binomial" if total > 30 else "Poisson")} error bars)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Number of Votes', fontsize=12)
    plt.xlabel('Preference Category', fontsize=12)
    
    # Add percentage annotations
    if total > 0:
        percentages = [f'({v/total*100:.1f}%)' for v in values]
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                    pct, ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f"{DATA_DIR}/comparison_chart_{filename.replace('.json', '.png')}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    
    plt.show()

def create_combined_chart(all_analyses):
    """Create a combined bar chart for all sessions."""
    if len(all_analyses) <= 1:
        return
    
    # Aggregate all results
    total_staircase = sum(a['staircase_wins'] for a in all_analyses)
    total_topk = sum(a['topk_wins'] for a in all_analyses)
    total_indifferent = sum(a['indifferent'] for a in all_analyses)
    total_comparisons = sum(a['total_comparisons'] for a in all_analyses)
    
    categories = ['Staircase Wins', 'TopK Wins', 'Indifferent']
    values = [total_staircase, total_topk, total_indifferent]
    colors = ['#2E8B57', '#CD5C5C', '#708090']
    
    # Calculate error bars (Poisson for small counts, Binomial for larger samples)
    error_bars = []
    
    for value in values:
        if total_comparisons > 30 and value > 5:  # Use binomial approximation for larger samples
            # Binomial standard error: sqrt(n * p * (1-p))
            p = value / total_comparisons if total_comparisons > 0 else 0
            error = np.sqrt(total_comparisons * p * (1 - p)) if total_comparisons > 0 else 0
        else:  # Use Poisson approximation for small counts
            # Poisson standard error: sqrt(count)
            error = np.sqrt(value) if value > 0 else 0
        error_bars.append(error)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='black',
                   yerr=error_bars, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add value labels on bars
    for bar, value, error in zip(bars, values, error_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.8,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title(f'Combined Blind Comparison Results\n'
              f'Total Comparisons: {total_comparisons} across {len(all_analyses)} sessions (with {("Binomial" if total_comparisons > 30 else "Poisson")} error bars)', 
              fontsize=16, fontweight='bold')
    plt.ylabel('Number of Votes', fontsize=14)
    plt.xlabel('Preference Category', fontsize=14)
    
    # Add percentage annotations
    if total_comparisons > 0:
        percentages = [f'({v/total_comparisons*100:.1f}%)' for v in values]
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                    pct, ha='center', va='center', fontweight='bold', 
                    color='white', fontsize=11)
    
    plt.tight_layout()
    
    # Save the combined plot
    output_path = f"{DATA_DIR}/comparison_chart_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined chart saved to: {output_path}")
    
    plt.show()

def print_detailed_summary(all_analyses):
    """Print detailed text summary of all results."""
    print("\n" + "="*60)
    print("DETAILED COMPARISON RESULTS")
    print("="*60)
    
    for analysis in all_analyses:
        print(f"\n👤 User: {analysis['user_name']}")
        print(f"📅 Session: {analysis['session_start']}")
        print(f"📊 Total comparisons: {analysis['total_comparisons']}")
        
        if analysis['total_comparisons'] > 0:
            total = analysis['total_comparisons']
            print(f"   🟢 Staircase wins: {analysis['staircase_wins']} ({analysis['staircase_wins']/total*100:.1f}%)")
            print(f"   🔴 TopK wins: {analysis['topk_wins']} ({analysis['topk_wins']/total*100:.1f}%)")
            print(f"   ⚪ Indifferent: {analysis['indifferent']} ({analysis['indifferent']/total*100:.1f}%)")
    
    if len(all_analyses) > 1:
        # Combined statistics
        total_staircase = sum(a['staircase_wins'] for a in all_analyses)
        total_topk = sum(a['topk_wins'] for a in all_analyses)
        total_indifferent = sum(a['indifferent'] for a in all_analyses)
        total_comparisons = sum(a['total_comparisons'] for a in all_analyses)
        
        print(f"\n{'='*30} COMBINED RESULTS {'='*30}")
        print(f"📊 Total comparisons across all sessions: {total_comparisons}")
        if total_comparisons > 0:
            print(f"   🟢 Total Staircase wins: {total_staircase} ({total_staircase/total_comparisons*100:.1f}%)")
            print(f"   🔴 Total TopK wins: {total_topk} ({total_topk/total_comparisons*100:.1f}%)")
            print(f"   ⚪ Total Indifferent: {total_indifferent} ({total_indifferent/total_comparisons*100:.1f}%)")

def main():
    """Main analysis function."""
    print("📊 Blind Comparison Results Analyzer")
    print("="*50)
    
    # Load all result files
    results = load_comparison_results()
    
    if not results:
        return
    
    print(f"Found {len(results)} result file(s)")
    
    # Analyze each session
    all_analyses = []
    for result in results:
        analysis = analyze_single_session(result['data'])
        all_analyses.append(analysis)
        
        print(f"\n📁 Processing: {result['filename']}")
        print(f"   User: {analysis['user_name']}")
        print(f"   Comparisons: {analysis['total_comparisons']}")
        
        # Create individual chart
        create_single_session_chart(analysis, result['filename'])
    
    # Create combined chart if multiple sessions
    create_combined_chart(all_analyses)
    
    # Print detailed summary
    print_detailed_summary(all_analyses)
    
    print("\n" + "="*60)
    print("✅ Analysis complete! Charts saved to data/ directory")
    print("="*60)

if __name__ == "__main__":
    main() 
# %%
