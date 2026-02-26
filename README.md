Hello,

This repo contains code and analysis output for our inflammation marker analysis of the PDP1 trial, 
where we tested psilocybin for the treatment of Parkinson's disease. Here is the main paper:
[Psilocybin therapy for mood dysfunction in Parkinson’s disease: an open-label pilot trial](https://www.nature.com/articles/s4138)

This analysis of blood markers is currently under review, will add link to published paper when 
accepted.

Notes:
- At this point we do not share data publicaly, if you would like to collaborate please reach out.
- Codebase works in tandem with the [szb_commons](https://github.com/szb37/szb_commons/tree/6ed6830e2a3a33c0ba2953369fc536ebb512b7a5) repo. Make sure you get the repo at the commit the link points to, backwards compatibility is not guaranteed. Once you have the repo on disk, change *path_szb_commons* in *src/config.py* to point to its location on your system.
- It is expected that you run the analysis from the conda environment (=conda_env.yml) included in this repo.
- analysis.ipynb walks you through every stats /tables / figure in the main apper. 