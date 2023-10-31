# Installation

## Docker container

Build the image using the dockerfile in docker/Dockerfile.

## Without docker

If you want to run the code without docker, you can find all the dependancise in the dockerfile in docker/Dockerfile.

# Extract knowledge neuron activations (fc1 actications)

* Launch the analyze_prompt.py script in /code. It will feed the LM with the templates found in [PROMPT FILE] and filled with the LAMA triplet.
* During the foward pass, various stats are gathered, including the fc1 activation (i.e. the knowledge neurons' activations), but also others such as micro accuracy or perplexity.
* It will output a 'fc1_data_*.pickle' file containing the extracted activations and the other stats.

> python code/analyze_prompts.py \
    --device cuda \
    --output_predictions False \
    --output_dir [WHERE TO OUTPUT THE ACTIVATION FILE] \
    --test_data_dir $DATA \
    --model_name [HUGGING FACE MODEL NAME] \
    --prompt_files [PROMPT FILE] \
    --eval_batch_size $BS \
    --common_vocab_filename $VOCAB \
    --relation $REL

* $VOCAB can be set to 'none' if you want to use the whole vocabulary (recommended)
* The prompt file has to be formatted as a json file, cf. this [exemple](https://drive.google.com/file/d/1yyTmAo2lgCTyhQ-xBUUrYtPp-LvcRFAU/view?usp=drive_link)
* $DATA: Our version of LAMA can be dowloaded [here](https://drive.google.com/file/d/1TWYjf_QWo-zn8ryjNW1eeJOMq9nE5HZL/view?usp=drive_link)
* In our study we used the [OPT family](https://huggingface.co/docs/transformers/model_doc/opt) of LMs. Using an other type of LM might require to modify the code.


# Quantitative analysis

* Once you have generated the 'fc1_data_*.pickle' files, you can use the following script to analyse it.
* You just have to specify the arguments which are direclty hard-coded in the script:

> MODEL='opt-350m'
> 
> EXP_NAME=f'{MODEL}'
> 
> SENSIBILITY_TRESHOLD=0
> 
> TRIGGER_TRESHOLD_FREQ_RATE=0.2
> 
> LOAD_FC1=[LIST OF fc1_data_*.pickle YOU WANT TO ANALYSE]

* Make sure the fc1_data_*.pickle matches with the model you are using.
* SENSIBILITY_TRESHOLD and TRIGGER_TRESHOLD_FREQ_RATE: you can use these default values
* Ideally, LOAD_FC1 contains fc1_data_*.pickle which have been extracted with different prompt types (e.g. Optiprompt, Autoprompt and Human)

Then, launch:

> python quantitative_analysis.py

* In the script, the data is formatted as a Pandas Dataframe see [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) if you are not familiar with it.

* 

# Qualitative analysis

In particular, we'll use this file:
OptiPrompt/code/exp_units.py

Here is an example of how to use it:
python3 exp_units.py --autoprompt --paraphrase --optiprompt --high_units --percentile_high 99 --n_units 100 --k_tokens 50 --save_dir [WHERE-TO-WRITE-THE-RESULTING-TXT] --fc1_datapath [PATH-OF-THE-FC1-DATA] --device cuda
In that case, it will launch the extraction of the 'high_units' (I'll tell you what it is later), for autoprompt paraphrase and optiprompt. It will extract 100 units per prompt type (the 100 units are randomly sampled from the set of high units), with 50 tokens associated to each unit. Here, the 'high_units" are defined as the units being activated more often than the 99 percentile (so they belong to the top 1%).

You can find the details of the function parameters at the beginning of the exp_units.py file (starting at line 24). But here are the main arguments:

Choose which prompt type you want to add in the experiment by adding/removing these options (if you want to extract shared or typical prompt, I recommend to only use two prompt types):
--optiprompt
--autoprompt
--paraphrase

Choose which units you want to extract:
--shared_units (units being highly activated for all prompt types)
--typical_units (units with high activation for the current type and low activation for the others)
--high_units (units with high activations)
--low_units (units with low activations)

The selection of high and low units is controlled by these parameters:
--percentile_high 90
--percentile_low 10

The selection of shared and typical units is controlled by these parameters:
--percentile_typical_max 80
--percentile_typical_min 20

So here is an other example. If you want to extract the typical, shared and high units for autoprompt and paraphrases:
python3 exp_units.py --autoprompt --paraphrase --shared_units --typical_units --high_units --percentile_high 99 --percentile_typical_max 80 --percentile_typical_min 20 --n_units 100 --k_tokens 50 --save_dir [WHERE-TO-WRITE-THE-RESULTING-TXT] --fc1_datapath [PATH-OF-THE-FC1-DATA] --device cuda
I'll not detail here how to launch the code on the cluster, but if you need help, ask me. It is based on the same libraries as we used before in this project. So you may re-use the same conda env (or whatever env you are using). You can also try to launch it locally on cpu (I am not sure if the GPUs are very useful for this code).
