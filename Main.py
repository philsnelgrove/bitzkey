'''
Created on Feb. 28, 2022

@author: mikdr
'''
LIMIT = 1000000

def fibonacci(old, new, seq):
    newer = old + new
    if newer < LIMIT:
        seq.append(newer)
        fibonacci(new, newer, seq)
    else:
        seq.append(f"(The next item would be {newer})")
    return seq

def main():
    print("Hello World!")
    print(f"Fibonacci sequence up to {LIMIT}:", *fibonacci(0, 1, [1]))

if __name__ == '__main__':
    main()