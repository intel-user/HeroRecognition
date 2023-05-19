# Hero Recognition
Recognition the hero present in the message bar of game League of Legends.
## 1. Run 
### 1.2 Download pretrained model
Download model following this link: https://drive.google.com/drive/folders/18jSOT6X1-enTYW7E3Lu0cL1aPKPO36I8?usp=sharing

Save model in folder `/model`

### 1.2 Run with script
Install environment
`sh scripts/prepare_env.sh`  
or run `python -m pip install -r requirement.txt`

Run predict
`sh scripts/run.sh`
or run `python predict.py --source "source file"`
## 2. Result
The result is saved at folder _result_. It is a txt file following format: "image file path"    "hero name"
