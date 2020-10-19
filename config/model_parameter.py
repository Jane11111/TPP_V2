import tensorflow.compat.v1 as tf


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
        self.flags.DEFINE_integer('THP_stack_num',1,'the number of multiple self-attention')
        self.flags.DEFINE_integer('THP_head_num',1,'the head number of mltihead attention')
        self.flags.DEFINE_integer('THP_Mk',16, 'Mk in THP')
        self.flags.DEFINE_integer('THP_Mv',16, 'Mv in THP')
        self.flags.DEFINE_integer('THP_M',64,'M in THP')
        self.flags.DEFINE_integer('THP_MH',256,'MH in THP')
        self.flags.DEFINE_integer('THP_Mi',256,'Mi in THP')


        # 随机梯度下降sgd
        self.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop,sgd*)')
        self.flags.DEFINE_float('decay_rate', 0.001, 'decay rate')
        self.flags.DEFINE_float('llh_decay_rate', 1, 'decay rate')
        # 最大梯度渐变到5
        self.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
        # 训练批次32
        self.flags.DEFINE_integer('train_batch_size', 512, 'Training Batch size')
        # 测试批次128
        self.flags.DEFINE_integer('test_batch_size', 512, 'Testing Batch size')
        # 最大迭代次数
        self.flags.DEFINE_integer('max_epochs', 500, 'Maximum # of training epochs')
        # 每100个批次的训练状态
        self.flags.DEFINE_integer('display_freq', 10, 'Display training status every this iteration')
        self.flags.DEFINE_integer('eval_freq', 500, 'Display training status every this iteration')
        self.flags.DEFINE_integer('max_len', 150, 'max len of attention')
        self.flags.DEFINE_integer('global_step', 100, 'global_step to summery AUC')

        # Runtime parameters
        self.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.8,
                                'Gpu memory use fraction, 0.0 for allow_growth=True')
        # date process parameters
        self.flags.DEFINE_boolean('is_training', True, 'train of inference')
        self.flags.DEFINE_string('type', "mimic_total", 'raw date type')
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



        #TODO 1:check cuda
        self.flags.DEFINE_string('cuda_visible_devices', '2', 'Choice which GPU to use')

        #TODO 2: check model

        # model & prepare the dataset
        self.flags.DEFINE_boolean('split_data', False, "if data is needed to be splitted")
        self.flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate')
        # self.flags.DEFINE_string('model_name', "Vallina_Gru", 'model name')
        # self.flags.DEFINE_string('model_name', "MTAM_only_time_aware_RNN", 'model name')
        # self.flags.DEFINE_string('model_name', "MTAM_TPP_wendy_time", 'model name')
        # self.flags.DEFINE_string('model_name', "MTAM_TPP_wendy_att_time", 'model name')
        # self.flags.DEFINE_string('model_name', "TANHP_v2", 'model name')
        self.flags.DEFINE_string('model_name', "TANHP_v3", 'model name')
        # self.flags.DEFINE_string('model_name', "THP", 'model name')
        # self.flags.DEFINE_string('model_name', "NHP", 'model name')
        # self.flags.DEFINE_string('model_name', "RMTPP", 'model name')
        # self.flags.DEFINE_string('model_name', "SAHP", 'model name')
        # self.flags.DEFINE_string('model_name', "HP", 'model name')
        # self.flags.DEFINE_string('model_name', "IHP", 'model name')

        #TODO 3: check data
        self.flags.DEFINE_string('data_name', 'conttime', 'the type of the dataset')

        #TODO 4: check loss

        # loss function
        # self.flags.DEFINE_string('loss', 'cross_entropy', 'the loss function ')
        # self.flags.DEFINE_string('loss', 'llh_ce', 'the loss function ')
        # self.flags.DEFINE_string('loss', 'log_likelihood', 'the loss function ')
        # self.flags.DEFINE_string('loss', 'no_cross_entropy', 'the loss function ')
        # self.flags.DEFINE_string('loss', 'no_rmse', 'the loss function ')
        # self.flags.DEFINE_string('loss', 'no_loglikelihood', 'the loss function ')
        self.flags.DEFINE_string('loss', 'all', 'the loss function ')

        self.flags.DEFINE_float('scale', 1, 'the type of the dataset')
        self.flags.DEFINE_integer('inner_sims_len', 5, 'samples of the inner part for calculating time')
        self.flags.DEFINE_integer('outer_sims_len', 5, 'samples of the outer part for calculating the time')




        # temporary point process
        self.flags.DEFINE_integer('type_num',5,"the number of event types")
        self.flags.DEFINE_integer('sims_len',10,'max number of samples')

        self.flags.DEFINE_string('integral_cal','MC','the method to calculate integral')

        self.flags.DEFINE_string('origin_train_name','train.pkl','origin train name')
        self.flags.DEFINE_string('origin_test_name','test.pkl','origin test name')
        self.flags.DEFINE_string('processed_train_name', 'train.txt', 'origin train name')
        self.flags.DEFINE_string('processed_test_name', 'test.txt', 'origin test name')

        #prepare data


        self.flags.DEFINE_string('in_data_root_path','D://Project/TPP_V2/data/origin_data/data_event/','the root path of the dataset')
        self.flags.DEFINE_string('out_data_root_path', 'D://Project/TPP_V2/data/training_testing_data/data_event/', 'the root path of the dataset')


    def get_parameter(self,type):

        # if self.flags.FLAGS.model_name == 'IHP':
        #     self.flags.FLAGS.learning_rate = 0.0001
        # elif self.flags.FLAGS.model_name == 'RMTPP':
        #     self.flags.FLAGS.learning_rate = 0.0001
        # elif self.flags.FLAGS.model_name == 'IHP':
        #     self.flags.FLAGS.learning_rate = 0.0001


        if type == 'twitter_retweet':
            self.flags.FLAGS.type_num = 3
            self.flags.FLAGS.sims_len = 3
            self.flags.FLAGS.max_seq_len = 20

            self.flags.FLAGS.origin_train_name = 'train_small_time.pkl'
            self.flags.FLAGS.origin_test_name = 'test_small_time.pkl'
            self.flags.FLAGS.processed_train_name = 'train_small_time.txt'
            self.flags.FLAGS.processed_test_name = 'test_small_time.txt'

            self.flags.FLAGS.in_data_root_path = "/home/zxl/project/TPP_V2/data/origin_data/data_retweet/"
            self.flags.FLAGS.out_data_root_path = "/home/zxl/project/TPP_V2/data/training_testing_data/data_retweet/"
            # self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_retweet/"
            # self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_retweet/"
        elif type == 'so':
            self.flags.FLAGS.type_num = 22
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 20

            self.flags.FLAGS.origin_train_name = 'train_small_time.pkl'
            self.flags.FLAGS.origin_test_name = 'test_small_time.pkl'
            self.flags.FLAGS.processed_train_name = 'train_small_time.txt'
            self.flags.FLAGS.processed_test_name = 'test_small_time.txt'

            self.flags.FLAGS.in_data_root_path = "/home/zxl/project/TPP_V2/data/origin_data/data_so/fold4/"
            self.flags.FLAGS.out_data_root_path = "/home/zxl/project/TPP_V2/data/training_testing_data/data_so/fold4/"
            # self.flags.FLAGS.in_data_root_path = "/Users/wendy/Documents/code/TPP_V2/data/origin_data/data_hawkes/"
            # self.flags.FLAGS.out_data_root_path = "/Users/wendy/Documents/code/TPP_V2/data/training_testing_data/data_hawkes/"
        elif type == 'hawkes':
            self.flags.FLAGS.type_num = 5
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 20
            self.flags.FLAGS.in_data_root_path = "/home/zxl/project/TPP_V2/data/origin_data/data_hawkes/"
            self.flags.FLAGS.out_data_root_path = "/home/zxl/project/TPP_V2/data/training_testing_data/data_hawkes/"
            # self.flags.FLAGS.in_data_root_path = "/Users/wendy/Documents/code/TPP_V2/data/origin_data/data_hawkes/"
            # self.flags.FLAGS.out_data_root_path = "/Users/wendy/Documents/code/TPP_V2/data/training_testing_data/data_hawkes/"

        elif type == 'hawkesinhib':
            self.flags.FLAGS.type_num = 5
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            self.flags.FLAGS.in_data_root_path = "/home/zxl/project/TPP_V2/data/origin_data/data_hawkesinhib/"
            self.flags.FLAGS.out_data_root_path = "/home/zxl/project/TPP_V2/data/training_testing_data/data_hawkesinhib/"
            # self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_hawkesinhib/"
            # self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_hawkesinhib/"
        elif type == 'conttime':
            self.flags.FLAGS.type_num = 5
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            self.flags.FLAGS.in_data_root_path = "/home/zxl/project/TPP_V2/data/origin_data/data_conttime/"
            self.flags.FLAGS.out_data_root_path = "/home/zxl/project/TPP_V2/data/training_testing_data/data_conttime/"
            # self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_conttime/"

        elif type == 'financial':
            self.flags.FLAGS.type_num = 2
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50

            self.flags.FLAGS.origin_train_name = 'train_small_time_for_rmtpp.pkl'
            self.flags.FLAGS.origin_test_name = 'test_small_time_for_rmtpp.pkl'
            self.flags.FLAGS.processed_train_name = 'train_small_time_for_rmtpp.txt'
            self.flags.FLAGS.processed_test_name = 'test_small_time_for_rmtpp.txt'


            # self.flags.FLAGS.in_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/origin_data/data_conttime/"
            # self.flags.FLAGS.out_data_root_path = "/home/cbd109/Users/zxl/PythonProject/TPP_V2/data/training_testing_data/data_conttime/"
            self.flags.FLAGS.in_data_root_path = "/home/zxl/project/TPP_V2/data/origin_data/data_bookorder/fold1/"
            self.flags.FLAGS.out_data_root_path = "/home/zxl/project/TPP_V2/data/training_testing_data/data_bookorder/fold1/"

        elif type == 'mimic_total':
            self.flags.FLAGS.type_num = 75
            # self.flags.FLAGS.max_length_seq = 100
            self.flags.FLAGS.max_seq_len = 50
            self.flags.FLAGS.in_data_root_path = "D://Project/TPP_V2/data/origin_data/data_mimic/total/"
            self.flags.FLAGS.out_data_root_path = "D://Project/TPP_V2/data/training_testing_data/data_mimic/total/"
            # self.flags.FLAGS.in_data_root_path = "/home/zxl/project/TPP_V2/data/origin_data/data_mimic/total/"
            # self.flags.FLAGS.out_data_root_path = "/home/zxl/project/TPP_V2/data/training_testing_data/data_mimic/total/"

        return  self.flags



