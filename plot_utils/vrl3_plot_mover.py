# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: plotting helper code is currently being cleaned up

from vrl3_plot_helper import *

# this maps a data folder name to several amlt exp folder names. for example,
data_folder_to_amlt_dict = {
    'distill': ['driven-bedbug', 'enabled-starfish', 'humorous-octopus', 'singular-treefrog', # distill with cap 1, num_c 32, change distill weight
                'fair-corgi', 'present-honeybee', # low number of channels
                'guiding-earwig', 'bright-piranha' # high cap ul encoder
                ],
    'n_compress':['coherent-caribou'], # change number of layer in the compression layers
    'double':['select-hamster', 'modest-guppy', 'accepted-dragon'], # double encoder design, performance not great
    'intermediate':['precious-hagfish', 'pleased-mosquito', 'resolved-cattle', 'selected-jawfish'], # single encoder, but try use intermediate layer output, see which layer gives best result
    'double-add':['sweet-glider'], # double resnet with additional conv output, see if additional output for double resnet can help
    'test-pg':['well-cow', 'optimum-corgi', 'robust-termite'], # double standard enc 8 variants, single res 8 variants. These results will help decide whether our hypothesis on policy gradient is correct or not
    'naivelarge':['liberal-macaque'], # naively do single encoder, rl update, deep encoder
    'newbaseline3s':['peaceful-anchovy', 'poetic-airedale', 'native-fawn', 'light-peacock', 'firm-starfish'],
    'newdouble3s':['precious-koi', 'central-louse',
                   'sound-asp', 'welcomed-doe', 'fancy-polecat',
                   'oriented-kid' # humanoid
                   ],
    'utd':['innocent-gar','trusting-warthog'],
    'pt': ['major-snipe', 'oriented-fox', 'brief-barnacle', 'superb-shepherd', 'golden-bobcat',
           'capital-alpaca', 'premium-ringtail', # stage 2
            ],
    'drq-base':['enabling-vervet',], # drq baseline with drq original code
    'tl': ['able-starling' ,# tl freeze on easy 6
           ],
    'ssl': ['on-gazelle',  'sincere-walrus', # freeze
                           'light-sheep', 'settled-mastiff', 'cheerful-muskrat'],
    's2-base':['needed-hedgehog'],
    'ssl-s2':['verified-impala', 'expert-cockatoo'],
    'adroit-firstgroup':['lasting-ferret', 'united-basilisk'],
    'adroit-secondgroup': ['upright-starling','star-opossum','distinct-viper','humorous-dassie',
                           'genuine-lynx', # no data aug
                           'choice-monkey', 'new-mole', # smaller lr
                           'quiet-worm', 'cuddly-cat' # sanity check
                           ],
    'adroit-stable': ['regular-monkfish', 'unique-monster',
                           ],
    'adroit-stable-hyper': ['loved-mastodon', 'equipped-owl', 'proven-urchin', 'innocent-pangolin'], # 1-seed hyper search
    'adroit-stable-hyper2': ['proven-urchin', 'innocent-pangolin'],
    'adroit-policy-lr': ['exciting-toad'],
    'adroit-s2cql':['composed-salmon','driven-reptile'],
    'adroit-s2cql2': ['adequate-reptile', 'native-kodiak', 'summary-baboon', 'safe-mammal'],
    'adroit-relocatehyper': ['dashing-grizzly'],
    'adroit-relocatehyper2': ['vast-robin'],
    'adroit-relocatehyper3': ['quality-gull', 'peaceful-elf'],
    'adroit-relocatehyper4': ['giving-krill', 'key-lizard'],
    'adroit-relocatehyper3-best-10seed': ['strong-impala'],
    'res6-pen': ['handy-redbird'],
    'relocate-abl1': ['easy-sloth', 'casual-sloth','engaged-reptile','national-killdeer',
                   'sweeping-lion','finer-pup','fine-civet','guiding-mule',
                   'sweet-cicada','enormous-malamute','special-louse'], # first set of ablation
    'abl2': ['funny-eel','unified-moose','clear-crappie','blessed-redfish','full-weevil','credible-rat','active-javelin'],
    'main':['dashing-platypus','magical-magpie','adapted-tuna','bursting-clam'],
    'main2':['trusting-condor','cosmic-baboon','composed-foxhound','large-goshawk'],
    'main3':['loved-anemone','e0113_main3.yaml','adapting-polecat','many-cheetah'],
    'main4': ['living-boxer'],
    'abl':[
        's1_pretrain',
           's2_bc_update','s2_cql_random','s2_cql_std','s2_cql_update','cql_weight','s2_enc_update',
           's3_bc_decay','s3_bc_weight','s3_std','s23_aug',
           's23_lr','s23_demo_bs','s23_enc_lr','s23_lr','s23_pretanh_weight','s23_safe_q_factor', # first round of ablations
           'fs_s2_bc', 'fs_s3_freeze_enc', 'fs_s3_buffer_new',
            's1_main_seeds', # 12 more seeds for main results #TODO add after we get that result, mainseeds2 currently not used
        's2_naive_rl', # stage 2 do naive RL updates, and shut down safe Q target
        's3_conservative', # stage 3 keep using conservative loss
        's23_q_threshold', # change Q threshold value #TODO maybe add to hyper sens?
        's23_pretanh_bc_limit', # TODO maybe appendix focus study?
        's23_pretanh_cql_limit', # TODO maybe appendix focus study?
        'special_drq', 'special_drq2', # drq baseline (4 envs) # TODO might need to add to main plots?
        'special_sdrq', 'special_sdrq2', # drq baseline with safe q target technique
        'fs_s3_freeze_extra', # use stage 1, freeze for stage 2,3, give intermediate level features # TODO what about this?
        'special_ferm', 'special_ferm_abl', # ferm baseline (improved baseline)
        'special_vrl3_ferm', # vrl3 + extra ferm updates in stage 2
        'special_ferm_plus', # ferm, but with safe Q target technique
        's23_bn', # bn finetune ablation # TODO focus study?
        's23_mom_enc_elr', 's23_mom_enc_safeq', 's1_model',  'frame_stack', # post sub new results, TODO 'action_repeat',
        'model_elr_new', 'q_min_new', # 'model_elr', 'q_min' # old jobs failed
        'moral-mudfish', 'cuddly-chicken', # these two are stride experiments
        'model_hs_elr3_new',  'model_hs_elr2_new', 'model_hs_elr1_new',
        'fs_enc_s2only', 'fs_enc_s3only', 's2_naive_rl_safe', 'ferm_and_s1',
    ],
    # neurips 2022 new
    'ablneurips':['f0728_bc_new_ablation_dhp', 'f0731_bc_new_ablation_r',
                   'f0728_rand_first_layer_dh', 'f0731_rand_first_layer_p',
                  'f0728_channel_mismatch_dh', 'f0731_channel_mismatch_p',
                  'f0731_channel_latent_flow_dh',  'f0731_channel_latent_flow_p',
                  'f0802_channel_latent_sub_dhp',
                  # '0728_channel_mismatch_r', 'f0728_rand_first_layer_r', 'f0728_channel_latent_flow_r',
                  'f0802_channel_latent_sub_dhp',
                  # 'f0806_rand_first_layer_fs1_r','f0806_channel_latent_sub_r','f0806_channel_mismatch_fs1_r','f0806_channel_latent_cat_r',
                  # 'f0808_byol',
                  'f0808_bc_new_ablation_dhp', 'f0808_bc_new_ablation_r',
                  'f0808_channel_latent_cat_r', 'f0808_channel_latent_sub_r', 'f0808_channel_mismatch_fs1_r', 'f0808_rand_first_layer_fs1_r',
                  'f0817_rand_first_layer_dhp_new', 'f0817_channel_latent_cat_dhp', 'f0817_channel_latent_sub_dhp',
                  'f0817_channel_mismatch_dhp', 'f0817_bc_only_withs1_r', 'f0817_bc_only_nos1_dhp', 'f0817_bc_only_nos1_r',
                  'f0902_highcap', 'f0902_bc_new_ablation_nos1_r', 'f0902_byol'
                  ],
    'ablph':['ph_s1_pretrain', 'ph_fs_s3_freeze_enc', 'ph_s2_cql_update', 'ph_s2_enc_update', 'ph_s2_naive_rl', 'ph_s3_conservative', 'ph_s23_enc_lr',
            'ph_s23_safe_q_factor', 'ph_s2_cql_random', 'ph_s2_cql_std', 'ph_s2_cql_weight', 'ph_s2_disable',
             'ph_s3_std', 'ph_s23_lr', 'ph_s23_q_threshold', 'ph_s3_bc_decay', 'ph_s3_bc_weight', 'ph_s23_aug', 'ph_s23_demo_bs',
             'ph_s23_pretanh_weight', 'ph_s2_bc_update',
             'ph_s2_bc', 'ph_s23_bn', 'ph_s23_frame_stack', 'ph_s1_model',
             ], # these are for pen and hammer
    'dmc':['dmc_vrl3_s3_medium', 'dmc_vrl3_s3_medium_pretrain_new', 'dmc_vrl3_s3_easy','dmc_vrl3_s3_easy_pretrain','dmc_vrl3_s3_hard_new','dmc_vrl3_s3_hard_pretrain_new',
           'dmc_e25k_medium', 'dmc_e25k_easy_hard'],
    'dmc_hyper':['dmc_e25k_medium_h1'], # hyper search # , 'dmc_e25k_hard_h1' dmc_e25k_medium_h2
    'dmc_hyper2':['dmc_e25k_medium_h2'],
    'dmc_hard_hyper1':['dmc_e25k_hard_h1'],
    'dmc_newhyper_3seed': ['dmc_e25k_medium_nh', 'dmc_e25k_easy_hard_nh', 'ferm_dmc_e25k_all', 'dmc_all_drqv2fd']

    # dmc_e25k_easy_hard_nh dmc_e25k_medium_nh # 3 seed full exp using new hyperparameters

    # tsne_relocate_nobc1 tsne_relocate_nobc2 # will just analyze on devnode

    ## this is command to download all
    # amlt results -I "*.csv" s1_pretrain s2_bc_update  s2_cql_random s2_cql_std s2_cql_update cql_weight s2_enc_update s3_bc_decay  s3_bc_weight s3_std  s23_aug s23_lr s23_demo_bs s23_enc_lr s23_lr s23_pretanh_weight   s23_safe_q_factor fs_s2_bc fs_s3_freeze_enc fs_s3_buffer_new s1_main_seeds

    # amlt results -I "*.csv" s2_naive_rl s3_conservative s23_q_threshold s23_pretanh_bc_limit s23_pretanh_cql_limit special_drq special_drq2 special_sdrq special_sdrq2 fs_s3_freeze_extra special_ferm special_ferm_abl  special_vrl3_ferm special_ferm_plus s23_bn

    # amlt results -I "*.csv"  s23_safe_q_factor fs_s2_bc fs_s3_freeze_enc fs_s3_buffer_new s1_main_seeds

    # TODO 0507: following is all new experiments we have run
    # amlt results -I "*.csv"  ph_s1_pretrain ph_fs_s3_freeze_enc ph_s2_cql_update ph_s2_enc_update ph_s2_naive_rl ph_s3_conservative ph_s23_enc_lr
    # ph_s23_safe_q_factor ph_s2_cql_random ph_s2_cql_std ph_s2_cql_weight ph_s2_disable
    # ph_s3_std ph_s23_lr ph_s23_q_threshold
    # ph_s3_bc_decay ph_s3_bc_weight ph_s23_aug ph_s23_demo_bs ph_s23_pretanh_weight

    # ferm_dmc_e25k_all fs_enc_s2only fs_enc_s3only dmc_all_drqv2fd #### s2_naive_rl_safe ferm_and_s1
}

# 0525 new: ph_s2_bc ph_s23_bn ph_s23_frame_stack ph_s1_model

move_data = True
move_keys = ['double', 'intermediate', 'double-add', 'test-pg', 'naivelarge']
move_keys = [ 'newbaseline3s', 'newdouble3s']
move_keys = [ 'double', 'intermediate', 'double-add',]
move_keys = [ 'pt']
move_keys = [ 'newbaseline3s', 'newdouble3s', 'pt']
move_keys = [ 'newbaseline3s', 'newdouble3s', 'pt', 'utd', 'drq-base', 'ssl', 'tl', 's2-base', 'ssl-s2']
move_keys = ['adroit-firstgroup', 'adroit-secondgroup', 'adroit-stable', 'adroit-stable-hyper', 'adroit-stable-hyper2', 'adroit-policy-lr']
move_keys = ['adroit-policy-lr']
move_keys = ['adroit-s2cql']
move_keys = ['adroit-relocatehyper2', 'adroit-relocatehyper3', 'res6-pen'] # hyper 3 is best
move_keys = ['adroit-relocatehyper4']
move_keys = ['adroit-relocatehyper3-best-10seed', 'relocate-abl1', 'abl2', 'main', 'main2','main3', 'main4']
move_keys = ['abl', 'dmc', 'dmc_hyper', 'dmc_hyper2', 'dmc_hard_hyper1', 'dmc_newhyper_3seed', 'ablph']
move_keys = ['ablneurips']
if move_data:
    for data_folder in move_keys:
        list_of_amlt_folder = data_folder_to_amlt_dict[data_folder]
        for amlt_folder in list_of_amlt_folder:
            move_data_from_amlt(amlt_folder, data_folder)

