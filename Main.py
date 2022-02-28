'''
Created on Feb. 28, 2022

@author: Phil Snelgrove
'''
VALUE_LIMIT = 1000000
ITERATION_LIMIT = 10

def fibonacci(count, old, new, seq):
    newer = old + new
    if newer < VALUE_LIMIT:
        count += 1
        seq.append("- %d" %(newer))
        fibonacci(count, new, newer, seq)
    else:
        seq.append(f"\n(The next item would be {newer})")
        seq.append(f"\n{count} steps total.")
    return seq

def main():
    print("Hello World!")
    print(f"Fibonacci sequence up to {VALUE_LIMIT}:", *fibonacci(1, 0, 1, [1]))

if __name__ == '__main__':
    main()