import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

column_names = "name,sequence,pred,score,cys,val,gly,leu,piz,ala,trp,ser,thr,alloile,asp,arg,lys,asn,gln,tyr,ile,phe,hpg,beta-ala,horn,salicylate,dap,glu,dpg,dhba,anthranilate,pro,hty,orn,his,beta-lys,ketoval,hse,24dab,fa,ketophe,pipecolate,23dab,pyruvate,asp-gamma,paba,gra".split(',')

result_dir = "/shanjunjie/ProteinMultiClass/visual/data-001"
scan_dir = "/shanjunjie/ProteinMultiClass/scan/result/csv001/domain"

cpu_num = 24
file_num = 374

def process_file(i):
    output_df = [pd.DataFrame(columns=column_names) for _ in range(20)]

    total_cnt = 0
    more_cnt = 0

    result_path = os.path.join(result_dir, f"bacteria.nonredundant_protein.{i}.protein.full.csv")
    scan_path = os.path.join(scan_dir, f"bacteria.nonredundant_protein.{i}.protein.csv")
    
    if not os.path.exists(result_path) or not os.path.exists(scan_path):
        return output_df, total_cnt, more_cnt
    
    result_df = pd.read_csv(result_path)
    scan_df = pd.read_csv(scan_path)
    
    for name in scan_df['name'].values.tolist():
        cnt = len(scan_df[scan_df['name'] == name])
        total_cnt += 1
        if cnt > 20:
            more_cnt += 1
            continue
        for scan_row in scan_df[scan_df['name'] == name].values.tolist():
            seq_content = scan_row[2]
            result_row = result_df[result_df['sequence'] == seq_content]
            if result_row.shape[0] == 0:
                continue
            if result_row["score"].values.tolist()[0] < 0.95:
                continue
            output_row = [name]+result_row.values.tolist()[0]

            output_df[cnt-1] = pd.concat([output_df[cnt-1], pd.DataFrame(dict(zip(column_names, output_row)), index=[0], columns=column_names)], ignore_index=True)
    print(f"Finished processing file {i}")
    return output_df.drop_duplicates(), total_cnt, more_cnt

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=cpu_num) as executor:
        results = executor.map(process_file, range(1, file_num+1))

    output_df_list, total_cnt_list, more_cnt_list = zip(*results)

    # Combine the results from different processes
    final_output_df = [pd.DataFrame() for _ in range(20)]
    for dfs_list in output_df_list:
        for i, df in enumerate(dfs_list):
            final_output_df[i] = pd.concat([final_output_df[i], df], ignore_index=True)
    
    final_total_cnt = sum(total_cnt_list)
    final_more_cnt = sum(more_cnt_list)

    print(f"Toal: {final_total_cnt}, more than 20: {final_more_cnt}")

    # Write the final results
    for i in range(20):
        final_output_df[i].to_csv(f"/shanjunjie/ProteinMultiClass/visual/filtered/have{i+1}domain.csv", index=False)
