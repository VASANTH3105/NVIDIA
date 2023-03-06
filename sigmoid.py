w1, w2, b, x1, x2 = [float(x) for x in input().split()]
import math
sigmoid = w1*x1 + w2*x2 + b
sigmoid = round(1/(1+math.exp(sigmoid*-1)),4)
print(sigmoid)

'''
Sample Input 
0 1 2 1 2
Sample Output 
0.9820
'''
