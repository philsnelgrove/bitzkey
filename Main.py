'''
Created on Feb. 28, 2022

@author: mikdr
'''

def fibonacci(old, new):
    if old > 1000:
        return
    newer = old + new
    print("%d " %(new))
    fibonacci(new, newer)
    return

def main():
    print("Hello World!")
    fibonacci(0, 1)

if __name__ == '__main__':
    main()