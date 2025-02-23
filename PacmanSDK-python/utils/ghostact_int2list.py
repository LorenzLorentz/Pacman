def ghostact_int2list(act_int):
    assert act_int<=124, "??????"
    act_list=[]
    while not act_int==0:
        act_list.append(act_int%5)
        act_int//=5
    while len(act_list)<3:
        act_list.append(0)
    return act_list

if __name__ == "__main__":
    act_int = 1 + 3*5 + 4*25
    print(ghostact_int2list(act_int))