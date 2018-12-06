import csv
with open('Training_examples.csv','r') as f:
    reader = csv.reader(f)
    dlist = list(reader)
h = [['0','0','0','0','0','0']]
for i in dlist:
     print(i)
     if i[-1] == "Yes":
        j = 0
        for x in i:
            if x != "Yes":
                if x != h[0][j] and h[0][j] == '0':
                    h[0][j] = x
                elif x != h[0][j] and h[0][j] != '0':
                    h[0][j] = '?'
                else:
                    pass
            j = j+1
print("Most specific hypothesis is:")
print(h)
