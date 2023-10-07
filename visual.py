import pandas as pd

with open("/shanjunjie/protein/dtbl_files/bacteria.nonredundant_protein.1.protein.dtbl", "r") as fp:
    for i, line in enumerate(fp):
        if i < 3:
            continue
        num = 0
        for ii, item in enumerate(reversed(line.split(' '))):
            if not item.isnumeric():
                continue
            num += 1
            if num == 1:
                end = int(item)
            if num == 2:
                start = int(item)
                break
        print(start, end)