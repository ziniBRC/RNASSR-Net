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

for dataset in ${dataset_list[*]}
do 
	echo -e "\ntraining ${dataset}..."
	python main_RNAGraph_graph_classification.py --dataset "${dataset}" --gpu_id 6 --config configs/RNAgraph_graph_classification_GCN_in_vivo_100k.json
	echo -e "Train finished"
done 
