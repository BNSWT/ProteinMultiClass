import os
import shutil

# 输入的文件夹路径
input_dir = '/shanjunjie/protein/fasta'
# 输出的文件夹路径的基础名
output_base_dir = '/shanjunjie/protein/fasta_split'
# 分割的文件夹数量
folder_count = 20  # 调整此数值以适应你的情况

# # 创建新的文件夹
# os.makedirs(output_base_dir, exist_ok=True)

# # 读取所有的.faa文件
# faa_files = [f for f in os.listdir(input_dir) if f.endswith('.faa')]
# total_files = len(faa_files)

# # 每个文件夹中的文件数
# file_num_per_folder = total_files // folder_count
# file_num_per_folder += 1 if total_files % folder_count != 0 else 0

# # 按文件数分配到各个文件夹
# for i, faa_file in enumerate(faa_files):
#     new_dir = os.path.join(output_base_dir, str(i // file_num_per_folder))
#     os.makedirs(new_dir, exist_ok=True)
#     shutil.copy2(os.path.join(input_dir, faa_file), os.path.join(new_dir, faa_file))

# 对每个文件夹，生成一个bash脚本
# for i in range(len(faa_files) // file_num_per_folder + (len(faa_files) % file_num_per_folder != 0)):
for i in range(folder_count):
    script_content = """cd /shanjunjie/ProteinMultiClass/scan
# Path to your database file
database_file="/shanjunjie/ProteinMultiClass/data/core_region.hmm"

# Path to .faa files
faa_dir="{}"

# Loop through each .faa file in the directory
for file in /shanjunjie/protein/"$faa_dir"/*.faa
do
    # Extract the directory and the base file name without extension
    dir=$(dirname "$file")
    base=$(basename "$file" .faa)

    # Construct the output file name
    output_file="/shanjunjie/ProteinMultiClass/scan/result/dtbl/${{base}}.dtbl"

    # Perform the hmmscan operation
    /shanjunjie/hmmer/bin/hmmscan --domE 1e-5 --domtblout "$output_file" --cpu 16 "$database_file" "$file" > log.txt
done""".format('fasta_split/{}'.format(i))
    with open('script/run_{}.sh'.format(i), 'w') as f:
        f.write(script_content)