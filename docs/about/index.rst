About RegVelo
-------------

Understanding cellular dynamics and regulatory interactions is crucial for decoding the complex processes that govern cell fate and differentiation. 
Traditional RNA velocity methods capture dynamic cellular transitions by modeling changes in spliced and unspliced mRNA but lack integration with gene regulatory networks (GRNs), omitting critical regulatory mechanisms underlying cellular decisions. 
Conversely, GRN inference techniques map regulatory connections but fail to account for the temporal dynamics of gene expression.

With RegVelo, developed by `Wang et al. (biorxiv, 2024) <https://www.biorxiv.org/content/10.1101/2024.12.11.627935v1>`_, 
the research gap is bridged through combining RNA velocity's temporal insights with a regulatory framework to model transcriptome-wide splicing kinetics informed by GRNs.
This extend current RNA velocity framework to a full mechanism model, allows to model more complicated development process.
Further, by coupling with CellRank `Weiler et al. (Nature Methods, 2024) <https://www.nature.com/articles/s41592-024-02303-9>`_, RegVelo expands its capabilities to include robust perturbation predictions, linking regulatory changes to long-term cell fate decisions. 
CellRank employs velocity-based state transition probabilities to predict terminal cell states, and RegVelo enhances this framework with GRN-informed splicing kinetics, 
enabling precise simulations of transcription factor (TF) knockouts. This synergy allows for the identification of lineage drivers and the prediction of cell fate changes upon genetic perturbations.

RegVelo's application
~~~~~~~~~~~~~~~~~~~~~
- estimate RNA velocity govarned by gene regulation.
- infer latent time to indicating cellular differentiation process.
- estimate velocity intrinsic and extrinsic uncertainty :cite:p:`gayoso2024deep`.
- estimate regulon perturbation effects via CellRank framework :cite:p:`lange2022cellrank, weiler2024cellrank`.

RegVelo model
~~~~~~~~~~~~~~~~~~~
RegVelo leverages deep generative modeling to infer splicing kinetics, transcription rates, and latent cellular time while integrating GRN priors derived from multi-omics data or curated databases. 
RegVelo incorporates cellular dynamic estimates by first encoding unspliced (*u*) and spliced RNA (*s*) readouts into posterior parameters of a low dimensional latent variable - the cell representation - with a neural network. 
An additional neural network takes samples of this cell representation as input to parameterize gene-wise latent time as in our previous model veloVI. 
We then model splicing dynamics with ordinary differential equations (ODEs) specified by a base transcription *b* and GRN weight matrix *W* , 
describing transcription and inferred by a shallow neural network, constant splicing and degradation rate parameters  *beta* and *gamma*  , respectively, 
and estimated cell and gene-specific latent times. Importantly, existing methods for inferring RNA velocity consider a set of decoupled one-dimensional ODEs for which analytic solutions exist, but RegVelo relies on the single, high-dimensional ODE

.. math::
    \begin{align} 
    \frac{\mathrm{d} u_{g}(t)}{\mathrm{d} t} =\alpha_{g}(t) - \beta_{g} u_{g}(t), \\
    \frac{\mathrm{d} s_{g}(t)}{\mathrm{d} t} = \beta_{g} u_{g}(t) - \gamma_{g} s_{g}(t),
    \end{align}

that is now coupled through gene regulation-informed transcription

.. math::
    \alpha_g = h \left( \left [ W s(t) +b \right ] _{g} \right)

where *g* indicates the gene and *h* is a non-linear activation function. 
We predict the gene and cell-specific spliced and unspliced abundances using a parallelizable ODE solver, 
as this new system does not pose an analytic solution anymore; compared to previous approaches, we solve all gene dynamics at once instead of sequentially for each gene independently of all others. 
The forward simulation of the ODE solver allows for computing the likelihood function encompassing all neural network and kinetic parameters. 
We assume that the predicted spliced and unspliced abundances are the expected value of the Gaussian likelihood of the observed dataset and use gradient-based optimization to update all parameters. 
After optimization, we define cell-gene-specific velocities as splicing velocities based on the estimated splicing and degradation rates and predicted spliced and unspliced abundance. 
Overall, RegVelo allows sampling predicted readouts and velocities from the learned posterior distribution.


Perturbation prediction
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://github.com/theislab/regvelo/blob/main/docs/_static/img/perturbation_overview_fig.png?raw=true
   :alt: RegVelo perturbation introduction
   :width: 600px

RegVelo is a generative model that couples cellular dynamics with regulatory networks. 
We can, thus, perform in silico counterfactual inference to test the cellular response upon unseen perturbations of a TF in the regulatory network: for a trained RegVelo model, 
we ignore regulatory effects of the TF by removing all its downstream targets from the GRN, i.e., depleting the regulon, and generate the perturbed velocity vector field. 
The dissimilarity between the original and perturbed cell velocities - the perturbation effect score - reflects the local changes on each cell induced by perturbations; we quantify this score with cosine dissimilarity.

RNA velocity describes a high dimensional vector field representing cellular change along the phenotypic manifold but lacks interpretability and quantifiable measures of the long-term cell behavior. 
We recently proposed CellRank to bridge this gap by leveraging gene expression and an estimated vector field to model cell state transitions through Markov chains and infer terminal cell states. 
For each terminal state identified, CellRank calculates the probability of a cell transitioning to this state - the fate probability - that allows us to predict the cell's future state. 
By combining RegVelo’s generative model with CellRank, we connect gene regulation with both local cell dynamics and long-term cell fate decisions, and how they change upon in silico perturbations. 
In the context of our perturbation analyses, we compare CellRank’s prediction of cell fate probabilities for the original and perturbed vector fields, 
to find enrichment (increased cell fate probability) or depletion (decreased cell fate probability) effects towards terminal states.

See `Wang et al. (biorxiv, 2024) <https://www.biorxiv.org/content/10.1101/2024.12.11.627935v1>`_ for a detailed description of the methods and applications on different biological systems.




