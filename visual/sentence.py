import pandas as pd
import os

column_names = "name,sequence,pred,score,cys,val,gly,leu,piz,ala,trp,ser,thr,alloile,asp,arg,lys,asn,gln,tyr,ile,phe,hpg,beta-ala,horn,salicylate,dap,glu,dpg,dhba,anthranilate,pro,hty,orn,his,beta-lys,ketoval,hse,24dab,fa,ketophe,pipecolate,23dab,pyruvate,asp-gamma,paba,gra".split(',')

result_dir = "/shanjunjie/ProteinMultiClass/visual/data-new"
scan_dir = "/shanjunjie/ProteinMultiClass/scan/result/csv/domain"

output_df = [pd.DataFrame(columns=column_names, index=None) for i in range(5)]

for i in range(1, 375):
    result_path = os.path.join(result_dir, f"bacteria.nonredundant_protein.{i}.protein.full.csv")
    scan_path = os.path.join(scan_dir, f"bacteria.nonredundant_protein.{i}.protein.csv")
    
    result_df = pd.read_csv(result_path)
    scan_df = pd.read_csv(scan_path)
    
    for name in scan_df['name'].values.tolist():
        cnt = len(scan_df[scan_df['name'] == name])
        if cnt > 5:
            continue
        for scan_row in scan_df[scan_df['name'] == name].values.tolist():
            seq_content = scan_row[2]
            result_row = result_df[result_df['sequence'] == seq_content]
            output_row = [name]+result_row.values.tolist()[0]

            output_df[cnt-1]=pd.concat([output_df[cnt-1], pd.DataFrame(dict(zip(column_names, output_row)), index=[0],columns=column_names)],ignore_index=True)
    break

cnt = 1
for df in output_df:
    df.to_csv(f"/shanjunjie/ProteinMultiClass/visual/csv/have{cnt}domain.csv")
    cnt += 1