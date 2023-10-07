#!/bin/bash
#Use this script with caution, 'right' outcome is not guarranteed!!!
#This script should be used under a Unix-like system(eg, Ubuntu) with BLAST, HMMER (including Easel miniapps), bedtools, python3 and perl installed.
#You need to run 'chmod +x ./PKSclusterfinder.sh' to get program executeble if this is the debut.
#intermediate--|-script
#              |-file
#Usage ./AdomainSubstratePredictor.sh PfamList(1) input_protein.faa(2) outputfolder(3) SeqDB(4) PfamKey(5)

#This is compiled by Chi Zhang @ Westlake University.

##################################
#00 quality control
##################################

#check if prameter number is right
if [ "$#" -ne 5 ]
then
  echo "Usage ./NRPSPredictorMain.sh PfamList.hmm(1) input_protein.fa(2) outputfolder(3) BlastDB(4) KeyfileList.hmm(5)"
  exit 1
fi

#check if PfamList(1) meets Pfam.hmm format

if [ `grep -c "^HMMER" $1` -eq 0 -o `grep -c "^HMMER" $1` -ne `grep -c "^//" $1` ] 
then
  echo "Please check .hmm file!"
  exit 1
fi

#check if input_protein.faa(2) meets fasta format

if [ `grep -c "^>" $2` -eq 0 ]
then
  echo "Please check .fa file!"
  exit 1
fi

#check the number of sequences in input_protein.faa(2)

echo "There are `grep -c "^>" $2` sequence(s) in submitted fasta file."


if [ `sed -e 's/^\(>[^[:space:]]*\).*/\1/' $2 | grep "^>" | sort -u | wc -l` -ne `sed -e 's/^\(>[^[:space:]]*\).*/\1/' $2 | grep -c "^>"` ]
then
  echo "Warning! There are duplicated ID(s) in fasta file and please remove duplication:"
  echo $(sed -e 's/^\(>[^[:space:]]*\).*/\1/' $2 | grep "^>" | uniq -d)
  exit 1
fi

#check if no sequence has A domain at all

trap 'rm -f "$TMPFILE"' EXIT

TMPFILE=$(mktemp) || exit 1
hmmfetch $1 AMP-binding > $TMPFILE

if [ `/shanjunjie/hmmer/bin/hmmsearch --domE 1e-5 $TMPFILE $2 | grep -c "\[No hits detected that satisfy reporting thresholds\]"` -ne 0 ]
then
  echo "Please submit protein sequence(s) containing A domain(s)!"
  exit 1
fi

#check if outputfolder exist

if [ ! -e $3 ]
then
  echo "Outputfolder doesn't exist!."
  echo 'Do you want to create it (please make sure you have the privilege in parent directory if choose Y!)? (Y/N):'
  read choice
  if [ "$choice" = 'Y' -o "$choice" = 'y' ]
  then
    mkdir -p $3
    echo "$3 was created"
  else
    echo "Then, please specify another folder!"
    exit 1
  fi
fi

#create working subfolder and fetch matrix

WORKDIR=$3/NRPSPredictor$$ #remove last/ avoid directory error in python

mkdir -p $WORKDIR/00_QC/

sed -e 's/^\(>[^[:space:]]*\).*/\1/' $2 > $WORKDIR/00_QC/00_RAWFASTA.fa

sed -ie 's/[[:space:]\/]/@/g' $WORKDIR/00_QC/00_RAWFASTA.fa

esl-sfetch --index $WORKDIR/00_QC/00_RAWFASTA.fa > /dev/null

##################################
#01 A domain (200AA) extraction and form a information table
##################################

mkdir $WORKDIR/01_DE/

hmmfetch $1 AdataSET0409200AA > $TMPFILE

/shanjunjie/hmmer/bin/hmmsearch --domT 41 --domtblout $WORKDIR/01_DE/200Ahmmsearch.dtbl --noali $TMPFILE $WORKDIR/00_QC/00_RAWFASTA.fa > /dev/null

grep -v "^#" $WORKDIR/01_DE/200Ahmmsearch.dtbl | awk '{print $1"/"$20"-"$21, $20, $21, $1}' | sort -t'/' -k1,1 -k2n,2 | esl-sfetch -Cf $WORKDIR/00_QC/00_RAWFASTA.fa - | awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' - | tail -n +2 > $WORKDIR/01_DE/ADomainhits.fa

grep ">" $WORKDIR/01_DE/ADomainhits.fa | awk -F'/' '{print $1}' - | sort -u > $WORKDIR/01_DE/ADomainhead.txt

grep ">" $WORKDIR/01_DE/ADomainhits.fa > $WORKDIR/01_DE/ADomainHIThead.txt

grep ">" $WORKDIR/00_QC/00_RAWFASTA.fa | sed -e 's/^\(>[^[:space:]]*\).*/\1/' | sort > $WORKDIR/01_DE/RAWhead.txt

comm -23 $WORKDIR/01_DE/RAWhead.txt $WORKDIR/01_DE/ADomainhead.txt > $WORKDIR/01_DE/NoADomainlist.txt

if [ `wc -l $WORKDIR/01_DE/NoADomainlist.txt | awk '{print $1}'` -ne 0 ] 
then
  echo "Warning! There is no A domain in the following sequences and they are not subject to upcoming analyses:"
  cat $WORKDIR/01_DE/NoADomainlist.txt
  echo
  echo
fi


##################################
#02 A domain identity search (hmmalign+esl-alipid) IDP
################################## 

echo "IDP MODE RUNNING ..."

mkdir -p $WORKDIR/02_HMMALIGN/

hmmalign --informat fasta --amino --outformat afa -o $WORKDIR/02_HMMALIGN/QUERYalignnoformat.fa $TMPFILE $WORKDIR/01_DE/ADomainhits.fa

awk '/^>/ {printf("\n%s\n",$0);next; } { printf("%s",$0);}  END {printf("\n");}' $WORKDIR/02_HMMALIGN/QUERYalignnoformat.fa | tail -n +2 > $WORKDIR/02_HMMALIGN/QUERYalign.fa

hmmalign --mapali $4 --informat fasta --amino --outformat afa -o $WORKDIR/02_HMMALIGN/QUERYalignwithCharacterized.fa $TMPFILE $WORKDIR/01_DE/ADomainhits.fa

esl-alipid --amino --informat afa $WORKDIR/02_HMMALIGN/QUERYalignwithCharacterized.fa > $WORKDIR/02_HMMALIGN/IDPlist.txt

cut -c2- $WORKDIR/01_DE/ADomainHIThead.txt > $WORKDIR/02_HMMALIGN/QUERYIDlist.txt

grep -Fwf $WORKDIR/02_HMMALIGN/QUERYIDlist.txt $WORKDIR/02_HMMALIGN/IDPlist.txt | awk '{print $1,$2,$3}' > $WORKDIR/02_HMMALIGN/TargetIDPlist.txt

cat $WORKDIR/02_HMMALIGN/TargetIDPlist.txt > $WORKDIR/02_HMMALIGN/IDPlist.txt # original IDPlist.txt is too large, so overwrite it with TargetIDPlist.txt after using

echo "
inputfile1 = open('$WORKDIR/02_HMMALIGN/TargetIDPlist.txt')
inputfile2 = open('$WORKDIR/02_HMMALIGN/QUERYIDlist.txt')
outputfile = open('$WORKDIR/02_HMMALIGN/IDPresult.txt', 'w')
line1 = inputfile1.readlines()
line2 = inputfile2.readlines()
inputfile1.close()
inputfile2.close()
INLIST=[]
OUTLIST=[]
SPACE='\t'
for m in line2:
  INLIST.append(m.strip())

for y in INLIST:
  NUM = 0
  TEMP = ''
  for x in line1:
    Ca, Cb, Cd = x.strip().split()
    Cc = float(Cd)
    if Ca == y and ( Cb not in INLIST ) and Cc > NUM:
        NUM = Cc
        TEMP = Ca+SPACE+Cb+SPACE+str(Cc)+'\n'
    elif Cb == y and ( Ca not in INLIST ) and Cc > NUM:
        NUM = Cc
        TEMP = Cb+SPACE+Ca+SPACE+str(Cc)+'\n'
    else:
        pass
  OUTLIST.append(TEMP)

for STR in OUTLIST:
    outputfile.write(STR)
outputfile.close()
" > $WORKDIR/02_HMMALIGN/IDPRANK.py
python3 $WORKDIR/02_HMMALIGN/IDPRANK.py

echo "IDP MODE DONE!"

##################################
#03 HMM whole sequence mode
##################################

echo "HMMwhole MODE RUNNING ..."

mkdir -p $WORKDIR/03_HMM/

echo -e 'qseqid\tHMMprofile\tBitscore\tlength\tE-value\tqstart\tqend' > $WORKDIR/03_HMM/TableHead

tail -n +1966 $1 > $WORKDIR/03_HMM/Onelayerforall.hmm

/shanjunjie/hmmer/bin/hmmsearch --domT 200 --domtblout $WORKDIR/03_HMM/NRPShmmsearch.dtbl --noali $WORKDIR/03_HMM/Onelayerforall.hmm $WORKDIR/01_DE/ADomainhits.fa > /dev/null

grep -v '^#' $WORKDIR/03_HMM/NRPShmmsearch.dtbl | awk '{print $1,$4,$8,$3,$7,$20,$21}' OFS='\t' | sort -k1,1 -k3nr,3 > $WORKDIR/03_HMM/ConciseHMMresult

echo "HMMwhole MODE DONE!"

##################################
#04 HMM key residues mode
##################################

echo "HMMkey MODE RUNNING ..."

mkdir -p $WORKDIR/04_KEYRES/

grep -v ">" $WORKDIR/02_HMMALIGN/QUERYalign.fa | cut -c7,19,21-23,25-39,46,48-66,82-84,100-104,123-128,130-133,136-137,139-140,149-153,155-156,159-163,177-180,183-189 > $WORKDIR/04_KEYRES/ONLYRES

sed -ie "s%[-\.]%%g" $WORKDIR/04_KEYRES/ONLYRES #remove "-" in alignment

grep ">" $WORKDIR/02_HMMALIGN/QUERYalign.fa > $WORKDIR/04_KEYRES/ONLYHEAD

paste $WORKDIR/04_KEYRES/ONLYHEAD $WORKDIR/04_KEYRES/ONLYRES | tr "\t" "\n" > $WORKDIR/04_KEYRES/EXTRACTKEYRES.fa

/shanjunjie/hmmer/bin/hmmsearch --domtblout $WORKDIR/04_KEYRES/KEYREShmmsearch.dtbl --noali $5 $WORKDIR/04_KEYRES/EXTRACTKEYRES.fa > /dev/null

grep -v '^#' $WORKDIR/04_KEYRES/KEYREShmmsearch.dtbl | awk '{print $1,$4,$8,$3,$7,$20,$21}' OFS='\t' | sort -k1,1 -k3nr,3 > $WORKDIR/04_KEYRES/ConciseKEYRESresult

echo "HMMkey MODE DONE!"

##################################
#05 Outputformatting
##################################

echo "Last step: formatting..."

mkdir -p $WORKDIR/05_Result/

cat $WORKDIR/02_HMMALIGN/IDPresult.txt > $WORKDIR/05_Result/IDPresult.txt

cat $WORKDIR/03_HMM/TableHead $WORKDIR/03_HMM/ConciseHMMresult > $WORKDIR/05_Result/HMMwholeresult.txt

cat $WORKDIR/03_HMM/TableHead $WORKDIR/04_KEYRES/ConciseKEYRESresult > $WORKDIR/05_Result/HMMKEYresult.txt

mkdir $WORKDIR/05_Result/Formatting

touch $WORKDIR/05_Result/Formatting/UniqueIDlist.txt

grep ">" $WORKDIR/01_DE/ADomainhits.fa | sort -t'/' -k1,1 -k2n,2 -u | cut -c2- > $WORKDIR/05_Result/Formatting/UAHEADlist.txt

for UniqueID in $(cut -c2- $WORKDIR/01_DE/ADomainhead.txt)
do
	if [ `grep -w -c $UniqueID $WORKDIR/05_Result/Formatting/UniqueIDlist.txt` -eq 0 ]
	then
		echo $UniqueID >> $WORKDIR/05_Result/Formatting/UniqueIDlist.txt
	fi
done



for ID in $(cat $WORKDIR/05_Result/Formatting/UniqueIDlist.txt)
do
  echo '>'$ID >> $WORKDIR/05_Result/NRPSPredictorResult.txt
  grep -w $ID $WORKDIR/05_Result/Formatting/UAHEADlist.txt > $WORKDIR/05_Result/Formatting/${ID}AdomainHEAD.txt
  
  for subID in $(cat $WORKDIR/05_Result/Formatting/${ID}AdomainHEAD.txt)
  do
    echo $subID >> $WORKDIR/05_Result/NRPSPredictorResult.txt

    echo "[HMMwhole]" >> $WORKDIR/05_Result/NRPSPredictorResult.txt
    #substrate=`grep -w -m 1 $subID $WORKDIR/05_Result/HMMwholeresult.txt | cut -f2 | grep -o '{.*}'`
    echo `grep -w -m 1 $subID $WORKDIR/05_Result/HMMwholeresult.txt | awk '{print $2,$3}'` >> $WORKDIR/05_Result/NRPSPredictorResult.txt

    echo "[HMMkey]" >> $WORKDIR/05_Result/NRPSPredictorResult.txt
    echo `grep -w -m 1 $subID $WORKDIR/05_Result/HMMKEYresult.txt | awk '{print $2,$3}'` >> $WORKDIR/05_Result/NRPSPredictorResult.txt

    echo "[IDP]" >> $WORKDIR/05_Result/NRPSPredictorResult.txt
    echo `grep -w -m 1 $subID $WORKDIR/05_Result/IDPresult.txt | awk '{print $2,$3}'` >> $WORKDIR/05_Result/NRPSPredictorResult.txt

    echo >> $WORKDIR/05_Result/NRPSPredictorResult.txt
  done
  echo >> $WORKDIR/05_Result/NRPSPredictorResult.txt
  echo >> $WORKDIR/05_Result/NRPSPredictorResult.txt
  echo >> $WORKDIR/05_Result/NRPSPredictorResult.txt
done

cat $WORKDIR/05_Result/NRPSPredictorResult.txt

echo "All done, cheer! Check result @ $WORKDIR/05_Result/NRPSPredictorResult.txt"

exit