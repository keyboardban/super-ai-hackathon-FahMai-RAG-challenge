import csv
import sys
import os

def load_csv(filename):
    data = {}
    try:
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    data[int(row[0])] = int(row[1])
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
    return data

def compare_submissions(file1, file2):
    print(f"Loading {file1}...")
    data1 = load_csv(file1)
    print(f"Loading {file2}...")
    data2 = load_csv(file2)
    
    if not data1 or not data2:
        print("Cannot compare. Missing data.")
        return
        
    common_ids = set(data1.keys()).intersection(set(data2.keys()))
    if not common_ids:
        print("No common questions found between the two files.")
        return
        
    match_count = 0
    diffs = []
    
    for qid in sorted(common_ids):
        ans1 = data1[qid]
        ans2 = data2[qid]
        if ans1 == ans2:
            match_count += 1
        else:
            diffs.append((qid, ans1, ans2))
            
    total = len(common_ids)
    match_rate = (match_count / total) * 100
    
    print("\n" + "="*60)
    print(f"COMPARISON RESULTS")
    print("="*60)
    print(f"File A: {file1}")
    print(f"File B: {file2}")
    print("-" * 60)
    print(f"Total Questions Compared : {total}")
    print(f"Matching Answers         : {match_count}")
    print(f"Different Answers        : {len(diffs)}")
    print(f"Agreement Rate           : {match_rate:.2f}%\n")
    
    if diffs:
        print("--- DIFFERENCES ---\n")
        print(f"{'Q ID':<6} | {'FILE A':<20} | {'FILE B':<20}")
        print("-" * 55)
        for qid, a1, a2 in diffs:
            print(f"Q {qid:<4} | Ans: {a1:<15} | Ans: {a2:<15}")

if __name__ == "__main__":
    f1 = sys.argv[1] if len(sys.argv) > 1 else "../submissions/advanced_submission.csv"
    f2 = sys.argv[2] if len(sys.argv) > 2 else "../submissions/ultimate_submission.csv"
    
    if not os.path.exists(f1) or not os.path.exists(f2):
        print(f"Please ensure both '{f1}' and '{f2}' exist.")
    else:
        compare_submissions(f1, f2)
