

### Data Download

Download [ESConv.json](https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json) and
put it in the folder `DATA`.

### Dara Preprocessing

Enter `DATA` and run ``python process.py``.

To preprocess the training data, run:


```console
python prepare.py  --config_name strat --inputter_name strat --train_input_file DATA/train.txt --max_input_length 160  --max_decoder_input_length 40

python prepare.py  --config_name strat_with_predict --inputter_name strat_with_predict --train_input_file DATA/train.txt --max_input_length 160  --max_decoder_input_length 40
```

### First Training Stage  

Run:


```console
. RUN/train_strat_predict.sh {gpu_id}
```


### Second Training Stage

Run:


```console
. RUN/train_strat_situation.sh {gpu_id} {model_path}
```
The ``{model_path}`` is the path of  model checkpoint folder after the first training stage.

### Third Training Stage

Change the value of ``--helpfulness_model_dir`` in the **RUN/align_strat** files.
The``--helpfulness_model_dir`` is the path of the helpfulness model checkpoint folder.
Run:


```console
. RUN/align_strat.sh {gpu_id} {model_path}
```

The ``{model_path}`` is the path of  model checkpoint folder after the second training stage.


The implementation the helpfulness_model's training process and the third training stage is based on the codes of the paper "Aligning language models with human preferences via a bayesian approach" (https://github.com/wangjs9/Aligned-dPM). We replace "mi_data.txt" with "esc_data.txt" to incorporate the impact of seekers' feedback scores, where the number of positive annotations
is equal to the seeker' feedback score.