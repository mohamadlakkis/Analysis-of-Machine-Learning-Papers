def longestUbrokenSublist(L):
    
    tempend=0
    end=0
    start=0
    tempstart=0
    for i in range(1,len(L)):

        if abs(L[i]-L[i-1])<=1:
            tempend+=1
            if tempend-tempstart> end-start:
                start=tempstart


                end=tempend
        else:
            tempstart=i
            tempend=i

    return L[start:end+1]






print(longestUbrokenSublist([]))
print(longestUbrokenSublist([2]))
print(longestUbrokenSublist([2,2]))
print(longestUbrokenSublist([1,0,3]))
print(longestUbrokenSublist([3,1,0]))
print(longestUbrokenSublist([1,1,-1,1,2]))
print(longestUbrokenSublist([1,1,-1,1,2,3]))
print(longestUbrokenSublist([1,1,-1,5,4,5,5,3,0,1,0]))
print(longestUbrokenSublist([0,5,1,1,0,1,0,7,1,2,5]))