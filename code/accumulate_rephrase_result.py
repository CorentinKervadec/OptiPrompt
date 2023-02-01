import pandas as pd
import os

# Put the different datafiles here:
datadic = {
    'facebook/opt-350m':{
        'dir':'output/relation-paraphrases_v2/facebook_opt-350m',
    },
}
output='prompts/marco_rephrasing'
# ---- utils
def read_tsv(path):
    # return a pandas dataframe extracted from the TSV
    return pd.read_csv(path, sep='\t', header=0)


def load_template_analyse(dir):
    data = []
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    for f in files:
        relation = f.split('.')[0]
        # reading the tsv
        dataframe = read_tsv(os.path.join(dir, f))
        # adding relation info to the dataframe
        dataframe['relation'] = [relation]*len(dataframe)
        data.append(dataframe)
    # concat all relations into one dataframe
    data = pd.concat(data)
    return data
# ---- \

if __name__ == "__main__":

    all_data = []
    # load all the data
    for model in datadic:
        dir = datadic[model]['dir']
        data = load_template_analyse(dir)
        # adding model info to the dataframe
        data['model'] = [model]*len(data)
        all_data.append(data)
    all_data = pd.concat(all_data)

    # to CSV
    all_data.to_csv(os.path.join(output,'rephrase_lama.csv'), header=True)
