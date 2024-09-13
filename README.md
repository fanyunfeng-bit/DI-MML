## Download data
- CREMAD: `https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz`
- AVE: `https://sites.google.com/view/audiovisualresearch`
- UCF:
`wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001`
`wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002`
`wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003`
`cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip`
`unzip ucf101_jpegs_256.zip`
`wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001`
`wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002`
`wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003`
`cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip`
`unzip ucf101_tvl1_flow.zip`
- ModelNet: `http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz`

## Preprocess data
- `video_preprocessing.py` in `./data/CREMAD` for CREMAD.
- `video_preprocessing.py` in `./data/AVE` for AVE.

## train
- CREMAD: 
`python main_CFT.py --dataset CREMAD --num_frame 3 --temperature 50 --train`
`python main_CFT_stage2.py --dataset CREMAD --num_frame 3 --load_path path_model1 --load_path_other path_model2 --temperature 1 --train`

- AVE: 
`python main_CFT.py --dataset AVE --num_frame 4 --temperature 10 --train`
`python main_CFT_stage2.py --dataset AVE --num_frame 4 --load_path path_model1 --load_path_other path_model2 --temperature 3 --train`

- UCF: 
`python main_CFT.py --dataset UCF --num_frame 1 --temperature 10 --train`
`python main_CFT_stage2.py --dataset UCF --num_frame 1 --load_path path_model1 --load_path_other path_model2 --temperature 1 --train`

- ModelNet: 
`python main_CFT.py --dataset ModelNet --num_frame 1 --temperature 50 --train`
`python main_CFT_stage2.py --dataset ModelNet --num_frame 1 --load_path path_model1 --load_path_other path_model2 --temperature 3 --train`

