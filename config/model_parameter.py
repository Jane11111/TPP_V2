import tensorflow as tf


class model_parameter:

    def __init__(self):
        # Network parameters
        self.flags = tf.flags
        self.flags.DEFINE_string('version', 'bpr', 'model version')
        self.flags.DEFINE_string('checkpoint_path_dir', 'data/check_point/bisIE_adam_blocks2_adam_dropout0.5_lr0.0001/','directory of save model')
        self.flags.DEFINE_integer('hidden_units', 8, 'Number of hidden units in each layer')
        self.flags.DEFINE_integer('num_blocks', 3, 'Number of blocks in each attention')
        self.flags.DEFINE_integer('num_heads', 1, 'Number of heads in each attention')
        self.flags.DEFINE_integer('num_units', 64, 'Number of units in each attention')
        self.flags.DEFINE_integer('type_emb_size', 64, 'Number of units in each attention')
        self.flags.DEFINE_list('layers',[],'layer units of intensity calculation network')
        self.flags.DEFINE_float('dropout', 0.1, 'Dropout probability(0.0: no dropout)')
        self.flags.DEFINE_float('regulation_rate', 0.00005, 'L2 regulation rate')

        # THP model parametre
        self.flags.DEFINE_integer('THP_stack_num',3,'the number of multiple self-attention')
        self.flags.DEFINE_integer('THP_head_num',3,'the head number of mltihead attention')
        self.flags.DEFINE_integer('THP_Mk',16, 'Mk in THP')
        self.flags.DEFINE_integer('THP_Mv',16, 'Mv in THP')
        self.flags.DEFINE_integer('THP_M',64,'M in THP')
        self.flags.DEFINE_integer('THP_MH',256,'MH in THP')
        self.flags.DEFINE_integer('THP_Mi',256,'Mi in THP')


        # 随机梯度下降sgd
        self.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop,sgd*)')
        self.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
        self.flags.DEFINE_float('decay_rate', 0.001, 'decay rate')
        # 最大梯度渐变到5
        self.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
        # 训练批次32
        self.flags.DEFINE_integer('train_batch_size', 1, 'Training Batch size')
        # 测试批次128
        self.flags.DEFINE_integer('test_batch_size', 32, 'Testing Batch size')
        # 最大迭代次数
        self.flags.DEFINE_integer('max_epochs', 500, 'Maximum # of training epochs')
        # 每100个批次的训练状态
        self.flags.DEFINE_integer('display_freq', 10, 'Display training status every this iteration')
        self.flags.DEFINE_integer('eval_freq', 500, 'Display training status every this iteration')
        self.flags.DEFINE_integer('max_len', 150, 'max len of attention')
        self.flags.DEFINE_integer('global_step', 100, 'global_step to summery AUC')

        # Runtime parameters
        self.flags.DEFINE_string('cuda_visible_devices', '2', 'Choice which GPU to use')
        self.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.8,
                                'Gpu memory use fraction, 0.0 for allow_growth=True')
        # date process parameters
        self.flags.DEFINE_boolean('is_training', True, 'train of inference')
        self.flags.DEFINE_string('type', "twitter_event", 'raw date type')
        self.flags.DEFINE_integer('max_seq_len',10,'the maximum length of history event')

        #parameters about origin_data
        self.flags.DEFINE_boolean('init_origin_data', False, 'whether to initialize the raw data')
        self.flags.DEFINE_boolean('init_train_data', False, 'whether to initialize the origin data')
        self.flags.DEFINE_integer('user_count_limit', 10000, "the limit of user")
        self.flags.DEFINE_string('causality', "unidirection", "the mask method")
        self.flags.DEFINE_string('pos_embedding', "time", "the method to embedding_beauty.csv pos")
        self.flags.DEFINE_integer('test_frac', 5, "train test radio")
        self.flags.DEFINE_float('mask_rate', 0.2, 'mask rate')
        self.flags.DEFINE_float('neg_sample_ratio', 20, 'negetive sample ratio')
        self.flags.DEFINE_boolean('remove_duplicate',True,'whether to remove duplicate entries')
        self.flags.DEFINE_string('experiment_data_type','item_based', 'item_based, dual')


        self.flags.DEFINE_string('fine_tune_load_path', None, 'the check point paht for the fine tune mode ')
        #parameters about model
        self.flags.DEFINE_string('load_type', "from_scratch", "the type of loading data")
        self.flags.DEFINE_boolean('draw_pic', False, "whether to draw picture")
        self.flags.DEFINE_integer('top_k', 20, "evaluate recall ndcg for k users")
        # self.flags.DEFINE_string('experiment_name', "data_init", "the expeiment")
        # self.flags.DEFINE_string('experiment_name', "MTAM", "the expeiment")

        # model & prepare the dataset
        # self.flags.DEFINE_string('model_name', "Vallina_Gru", 'model name')
        # self.flags.DEFINE_string('model_name', "MTAM_only_time_aware_RNN", 'model name')
        self.flags.DEFINE_string('model_name', "MTAM_TPP_wendy", 'model name')
        # self.flags.DEFINE_string('model_name', "THP", 'model name')

        # loss function
        # self.flags.DEFINE_string('loss', 'cross_entropy', 'the loss function ')
        # self.flags.DEFINE_string('loss', 'llh_ce', 'the loss function ')
        # self.flags.DEFINE_string('loss', 'log_likelihood', 'the loss function ')
        self.flags.DEFINE_string('loss', 'llh_ce_se', 'the loss function ')


        # temporary point process
        self.flags.DEFINE_integer('type_num',5,"the number of event types")
        self.flags.DEFINE_integer('sims_len',10,'max number of samples')
        self.flags.DEFINE_string('integral_cal','NU','the method to calculate integral')


        #prepare data
        self.flags.DEFINE_boolean('split_data',False, "if data is needed to be splitted")
        self.flags.DEFINE_string('data_name','mimic_fold3','the type of the dataset')

        self.flags.DEFINE_string('in_data_root_path','D://Project/TPP_V2/data/origin_data/data_event/','the root path of the dataset')
        self.flags.DEFINE_string('out_data_root_path', 'D://Project/TPP_V2/data/training_testing_data/data_event/', 'the root path of the dataset')


    def get_parameter(self,type):

        if self.flags.FLAGS.integral_cal == 'NU':
            self.flags.FLAGS.sims_len = 2

        if type == 'twitter_event':
            self.flags.FLAGS.type_num = 3
            # self.flags.FLAGS.max_length_seq = 3494
            self.flags.FLAGS.max_seq_len = 50
            self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_hawkes/"
            self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_hawkes/"
        elif type == 'twitter_retweet':
            self.flags.FLAGS.type_num = 3
            # self.flags.FLAGS.max_length_seq = 264
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_retweet/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_retweet/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_retweet/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_retweet/"

        elif type == 'hawkes':
            self.flags.FLAGS.type_num = 5
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 20
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_hawkes/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_hawkes/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_hawkes/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_hawkes/"
        elif type == 'hawkesinhib':
            self.flags.FLAGS.type_num = 5
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_hawkesinhib/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_hawkesinhib/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_hawkesinhib/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_hawkesinhib/"

        elif type == 'conttime':
            self.flags.FLAGS.type_num = 5
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_conttime/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_conttime/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_conttime/"
        elif type == 'mimic_fold1':
            self.flags.FLAGS.type_num = 75
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_conttime/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_mimic/fold1/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_mimic/fold1/"
        elif type == 'mimic_fold2':
            self.flags.FLAGS.type_num = 75
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_conttime/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_mimic/fold2/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_mimic/fold2/"
        elif type == 'mimic_fold3':
            self.flags.FLAGS.type_num = 75
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_conttime/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_mimic/fold3/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_mimic/fold3/"
        elif type == 'mimic_fold4':
            self.flags.FLAGS.type_num = 75
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_conttime/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_mimic/fold4/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_mimic/fold4/"
        elif type == 'mimic_fold5':
            self.flags.FLAGS.type_num = 75
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_conttime/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_mimic/fold5/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_mimic/fold5/"
        elif type == 'mimic_total':
            self.flags.FLAGS.type_num = 75
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_conttime/"
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_mimic/total/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_mimic/total/"

        return  self.flags



