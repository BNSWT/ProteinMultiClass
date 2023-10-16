cd /shanjunjie/ProteinMultiClass/convert_csv

# Loop through each .faa file in the directory
for file in "/shanjunjie/protein/dtbl_files"/*.dtbl
do
    # Extract the directory and the base file name without extension
    dir=$(dirname "$file")
    base=$(basename "$file" .dtbl)

    dbtl_path="$file"
    fasta_path="/shanjunjie/protein/fasta/${base}.faa"
    output_full_path="/shanjunjie/ProteinMultiClass/convert_csv/csv/hmm_result/full/${base}.csv"
    output_seq_path="/shanjunjie/ProteinMultiClass/convert_csv/csv/hmm_result/seq/${base}.csv"
    output_domain_path="/shanjunjie/ProteinMultiClass/convert_csv/csv/hmm_result/domain/${base}.csv"
    inference_dataset="/shanjunjie/ProteinMultiClass/convert_csv/csv/hmm_result/seq/${base}.csv"
    part_result_path="/shanjunjie/ProteinMultiClass/convert_csv/csv/domain_result/${base}.part.csv"
    full_result_path="/shanjunjie/ProteinMultiClass/convert_csv/csv/domain_result/${base}.full.csv"

    python convert_csv.py \
        --dbtl_path "$dbtl_path" \
        --fasta_path "$fasta_path" \
        --output_full_path "$output_full_path" \
        --output_seq_path "$output_seq_path" \
        --output_domain_path "$output_domain_path"

    python csv_inference.py \
        --inference_dataset "$inference_dataset" \
        --part_result_path "$part_result_path" \
        --full_result_path "$full_result_path"
done