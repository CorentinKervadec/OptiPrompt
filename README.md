I'm using the code from Optiprompt has it is more convenient when using optiprompt and autoprompt prompts, but most of the code we are using is new.

(To go on the eval branch do: 'git checkout eval'

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
