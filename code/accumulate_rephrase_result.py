import sys
sys.path.insert(0, '/homedtcl/ckervadec/OptiPrompt')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from exp_infos import LM_MODELS
import json
# Put the different datafiles here:
datadic = {
    'facebook/opt-350m':{
        'dir':'/homedtcl/ckervadec/experiments/results/select-rephrase-facebook-opt-350m.271063/relation-paraphrases_v2/facebook_opt-350m',
    },
    'facebook/opt-1.3b':{
        'dir':'/homedtcl/ckervadec/experiments/results/select-rephrase-facebook-opt-1.3b.271064/relation-paraphrases_v2/facebook_opt-1.3b'
    },
    'facebook/opt-6.7b':{
        'dir':'/homedtcl/ckervadec/experiments/results/select-rephrase-facebook-opt-6.7b.271065/relation-paraphrases_v2/facebook_opt-6.7b'
    },
    'facebook/opt-iml-max-1.3b':{
        'dir':'/homedtcl/ckervadec/experiments/results/select-rephrase-facebook-opt-iml-max-1.3b.271066/relation-paraphrases_v2/facebook_opt-iml-max-1.3b'
    },
    # 'gpt2-xl':{
    #     'dir':'/homedtcl/ckervadec/experiments/results/select-template-gpt2-xl.268079/relation-paraphrases_v2/gpt2-xl'
    # },
    # 'roberta-large':{
    #     'dir':'/homedtcl/ckervadec/experiments/results/select-template-roberta-large.268199/relation-paraphrases_v2/roberta-large'
    # },
    # 't5-large':{
    #     'dir':'/homedtcl/ckervadec/experiments/results/select-template-t5-large.268201/relation-paraphrases_v2/t5-large'
    # },
    # 'bert-large-cased':{
    #     'dir':'/homedtcl/ckervadec/experiments/results/select-template-bert-large-cased.268200/relation-paraphrases_v2/bert-large-cased'
    # },
}
output='../data/prompts/marco_rephrasing'
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
        data['model'] = [model.split('/')[-1]]*len(data)
        data['family'] = [LM_MODELS[model]['family']]*len(data)
        all_data.append(data)
    all_data = pd.concat(all_data).reset_index()

    # micro -> %
    all_data['micro'] = all_data['micro']*100

    # to CSV
    all_data.sort_values(by=['model', 'micro'], ascending=False).to_csv(os.path.join(output,'rephrase_lama.tsv'), header=True, sep='\t')
    all_data[all_data['format']=='S'][['model', 'relation', 'template', 'micro']].sort_values(by=['model', 'relation', 'micro'], ascending=False).to_csv(os.path.join(output,'rephrase_lama_opt_S.tsv'), header=True, sep='\t')


    def get_best_template(df):
        # Return best templates for each model
        idx_best_template = df.groupby(by=['relation', 'model'])['micro'].transform(max)==df["micro"]
        data_best_template = df[idx_best_template][["model","relation","id","template","micro"]]
        # Some models+relations have equally good best templates.
        # we want to keep the template with is also good on other models
        template_occurence = data_best_template['id'].value_counts()
        # new column with template occurence
        data_best_template['template_occurence'] = template_occurence[data_best_template['id']].values
        # drop duplicate and keep template with highest cpt
        # beforehand I sort by template alphabetic order, to solve extreme cases
        data_best_template = data_best_template.sort_values(by=['template_occurence', 'template'], ascending=False).drop_duplicates(["model", "micro"], keep='first')
        return data_best_template


    def save_json(df, savepath):
        dic_list = list(df[['relation', 'template']].to_dict('index').values())
        with open(savepath, 'w') as f:
            for d in dic_list:
                json.dump(d,f)
                f.write('\n')

    data_best_template = get_best_template(all_data)
    print(data_best_template)
    data_best_template.to_csv(os.path.join(output,'best_templates_QS.tsv'),
        columns=['model', 'relation', 'template', 'micro', 'template_occurence'],header=True, sep='\t')
    
    # # Save a sentence only version
    data_best_template_S = get_best_template(all_data[all_data['format']=='S'])
    data_best_template_S.to_csv(os.path.join(output,'best_templates_S.tsv'),
        columns=['model', 'relation', 'template', 'micro', 'template_occurence'],header=True, sep='\t')

    for m in datadic:
        model_name = m.split('/')[-1]
        save_json(data_best_template[data_best_template['model']==model_name], os.path.join(output, f'best_rephrase_{model_name}_QS.jsonl'))
        save_json(data_best_template_S[data_best_template_S['model']==model_name], os.path.join(output, f'best_rephrase_{model_name}_S.jsonl'))

    
    
    # gb = all_data.groupby(by=['relation', 'model'])
    # print(gb.loc[gb["micro"]==gb['micro'].max()])

    # print(all_data.loc[all_data.reset_index().groupby(by=['relation'])["micro"].idxmax()])

    #·· viz

    #· max / var
    # heatmap max
    # fig = plt.figure()
    # idx_best_template = all_data.groupby(by=['relation', 'model'])['micro'].transform(max)==all_data["micro"]
    # data_best_template = all_data[idx_best_template]
    # # sort and remove duplicate -> already done
    # # data_best_template =  data_best_template.sort_values("template").drop_duplicates(["model", "micro"], keep='first') # keep the first template given the alphabetic order
    # sns.clustermap(data_best_template.pivot("model", "relation", "micro"))
    # plt.tight_layout()
    # plt.savefig('max_heatmap.png', dpi=fig.dpi)

    # bar plot micro avg of max template
    fig = plt.figure()
    max_sorted_order = data_best_template.groupby(by=["model"])["micro"].mean().sort_values().iloc[::-1].index
    ax = sns.boxplot(data_best_template, x="model", y="micro", order=max_sorted_order)
    ax.set(ylabel='Avg micro w/ best template')
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('max_box.png', dpi=fig.dpi)
    
    # control for template format: Q or S
    fig = plt.figure()
    idx_best_template_fmt = all_data.groupby(by=['relation', 'model', "format"])['micro'].transform(max)==all_data["micro"]
    data_best_template_fmt = all_data[idx_best_template_fmt]
    data_best_template_fmt =  data_best_template_fmt.sort_values("template").drop_duplicates(["model", "format", "micro"], keep='first') # keep the first template given the alphabetic order
    # print(data_best_template_fmt)
    ax = sns.boxplot(data_best_template_fmt, x="model", y="micro", hue="format", order=max_sorted_order)
    ax.set(ylabel='Avg micro w/ best template')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('max_box_fmt.png', dpi=fig.dpi)

    #· consistency
    # boxplot
    fig = plt.figure()
    # compute std accross template inside each relation
    data_std_template = all_data.groupby(by=["relation","model"])["micro"].std().reset_index()
    sorted_order = data_std_template.groupby(by=["model"])["micro"].mean().sort_values().iloc[::-1].index
    ax = sns.boxplot(data_std_template, x="model", y="micro", order=sorted_order)
    ax.set(ylabel='Std accross templates')
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('var_box.png', dpi=fig.dpi)