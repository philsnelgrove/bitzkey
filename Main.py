'''
Created on Feb. 28, 2022

@author: mikdr
'''

def fibonacci(old, new, seq):
    newer = old + new
    if newer < 10000:
        seq.append(newer)
        fibonacci(new, newer, seq)
    return seq

def main():
    print("Hello World!")
    print("Fibonacci sequence up to 10,000:", *fibonacci(0, 1, []))

if __name__ == '__main__':
    main()