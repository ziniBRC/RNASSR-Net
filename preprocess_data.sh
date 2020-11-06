dataset_list=(
"ALKBH5_Baltz2012"
"C17ORF85_Baltz2012"
"C22ORF28_Baltz2012"
"CAPRIN1_Baltz2012"
"CLIPSEQ_AGO2"
"CLIPSEQ_ELAVL1"
"CLIPSEQ_SFRS1"
"ICLIP_HNRNPC"
"ICLIP_TDP43"
"ICLIP_TIA1"
"ICLIP_TIAL1"
"PARCLIP_AGO1234"
"PARCLIP_ELAVL1"
"PARCLIP_ELAVL1A"
"PARCLIP_EWSR1"
"PARCLIP_FUS"
"PARCLIP_HUR"
"PARCLIP_IGF2BP123"
"PARCLIP_MOV10_Sievers"
"PARCLIP_PUM2"
"PARCLIP_QKI"
"PARCLIP_TAF15"
"PTBv1"
"ZC3H7B_Baltz2012"
)

count=0
DIR=/data1/zini/benchmarking-gnns-master12/data/GraphProt_CLIP_sequences/RNAGraphProb/
for dataset in ${dataset_list[*]}
do 
	cd $DIR
	FILE="${dataset}.pkl"
	if test -f "$FILE"; then
		echo -e "$FILE already exists."
	else
		echo -e "\nprocessing $FILE..."
		cd ../../../
    pwd
		python preprocess.py "${dataset}" True
		echo -e "$FILE already exists."
	fi
done 
