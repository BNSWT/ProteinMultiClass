import csv
import argparse
import pandas as pd
from Bio import SeqIO

def convert(input_dbtl_path, output_full_path):
    # 打开输出CSV文件以写入数据
    with open(output_full_path, 'w', newline='') as csv_file:
        # 创建CSV写入器
        csv_writer = csv.writer(csv_file)
        
        # 写入CSV文件的标题行
        csv_writer.writerow(['Target', 'Query', 'Full E-value', 'Full Score', 'Full Bias', 'Domain C-Evalue', 'Domain I-Evalue', 'Domain Score', 'Domain Bias', 'HMM Start', 'HMM End', 'Ali Start', 'Ali End', 'Env Start', 'Env End', 'Acc', 'Description'])
        
        # 打开HMMscan结果文件以读取数据
        with open(input_dbtl_path, 'r') as file:
            # 遍历每一行
            for line in file:
                # 忽略注释行以及空行（可选）
                if line.startswith('#') or line.strip() == '':
                    continue
                
                # 分割行，通常使用空格或制表符作为分隔符
                fields = [item for item in line.split() if item!='']
                
                # 根据HMMscan结果文件的格式，提取需要的信息
                target_name = fields[0]
                seq_name = fields[3]
                                
                full_e_value = fields[6]
                full_score = fields[7]
                full_bias = fields[8]
                domain_c_e_value = fields[11]
                domain_i_e_value = fields[12]
                domain_score = fields[13]
                domain_bias = fields[14]
                hmm_start = fields[15]
                hmm_end = fields[16]
                ali_start = fields[17]
                ali_end = fields[18]
                env_start = fields[19]
                env_end = fields[20]
                acc = fields[21]
                description = ' '.join(fields[22:])
                
                # 将提取的信息写入CSV文件
                csv_writer.writerow([target_name, seq_name, full_e_value, full_score, full_bias, domain_c_e_value, domain_i_e_value, domain_score, domain_bias, hmm_start, hmm_end, ali_start, ali_end, env_start, env_end, acc, description])

    print(f"转换完成，结果已保存到 {output_full_path}")

def get_seq(output_full_path, input_fasta_path, output_seq_path):
    records = list(SeqIO.parse(input_fasta_path, "fasta"))
    # 打开输出CSV文件以写入数据
    with open(output_seq_path, 'w', newline='') as csv_file:
        # 创建CSV写入器
        csv_writer = csv.writer(csv_file)
        
        # 写入CSV文件的标题行
        csv_writer.writerow(['name', 'sequence'])
        
        # 打开HMMscan结果文件以读取数据
        df = pd.read_csv(output_full_path)
        seq_names = df['Query'].values.tolist()
        seq_names = set(seq_names)
        for seq_name in seq_names:
            for record in records:
                sequence_name = record.id
                sequence = record.seq
                if sequence_name == seq_name:
                    seq = sequence
                    break
                
            # 将提取的信息写入CSV文件
            csv_writer.writerow([seq_name, seq])

    print(f"转换完成，结果已保存到 {output_seq_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbtl_path', type=str, default='/shanjunjie/protein/dtbl_files/bacteria.nonredundant_protein.1.protein.dtbl')
    parser.add_argument('--fasta_path', type=str, default='/shanjunjie/protein/fasta/bacteria.nonredundant_protein.1.protein.faa')
    parser.add_argument('--output_full_path', type=str, default='full.csv')
    parser.add_argument("--output_seq_path", type=str, default='seq.csv')
    args = parser.parse_args()
    convert(args.dbtl_path, args.output_full_path)
    get_seq(args.output_full_path, args.fasta_path, args.output_seq_path)
