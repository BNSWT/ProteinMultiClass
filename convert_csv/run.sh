cd /shanjunjie/ProteinMultiClass/convert_csv
python convert_csv.py \
    --dbtl_path /shanjunjie/protein/dtbl_files/bacteria.nonredundant_protein.1.protein.dtbl \
    --fasta_path /shanjunjie/protein/fasta/bacteria.nonredundant_protein.1.fasta \
    --output_full_path /shanjunjie/ProteinMultiClass/convert_csv/csv/hmm_result/full/bacteria.nonredundant_protein.1.csv \
    --output_seq_path /shanjunjie/ProteinMultiClass/convert_csv/csv/hmm_result/seq/bacteria.nonredundant_protein.1.csv
python csv_inference.py \
    --inference_dataset /shanjunjie/ProteinMultiClass/convert_csv/csv/hmm_result/seq/bacteria.nonredundant_protein.1.csv