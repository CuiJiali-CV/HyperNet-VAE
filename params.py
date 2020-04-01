# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data: 
import numpy as np


class GeneratorParams:
    def __init__(self, category, img_size, z_size_i):
        if category == 'Mnist':
            # fc1 :
            self.fc1_filter_size = [z_size_i, 500]
            # fc2 :
            self.fc2_filter_size = [500, 500]
            # fc3 :
            self.fc3_filter_size = [500, 784]

class HyperNetParams:
    def __init__(self, category, img_size, z_size_h, z_size_i):
        paramofGen = GeneratorParams(category=category, img_size=img_size, z_size_i=z_size_i)
        if category == 'Mnist':
            """
            weights
            """
            self.w_gen_hiddenlayer_size = 2
            # [500,15]
            self.code1_size = [paramofGen.fc1_filter_size[1], 15]
            # [500,15]
            self.code2_size = [paramofGen.fc2_filter_size[1], 15]
            # [784, 15]
            self.code3_size = [paramofGen.fc3_filter_size[1], 15]


            # gen fc1 filter code[1, 1024, 15] - [1, 1024, 40] - [1, 1024, 40] - [1, 1024, z+10+1] - [z+10+1, 1024]
            self.fc1_w1_size = [self.code1_size[1], 40]
            self.fc1_w2_size = [40, 40]
            self.fc1_w3_size = [40, paramofGen.fc1_filter_size[0] + 1]

            # gen fc2 filter code[1, 6272, 15] - [1, 6272, 40] - [1, 6272, 40] - [1, 6272, 1024+10+1] - [1024+10+1, 6272]
            self.fc2_w1_size = [self.code2_size[1], 40]
            self.fc2_w2_size = [40, 40]
            self.fc2_w3_size = [40, paramofGen.fc2_filter_size[0] + 1]

            # gen dc1 filter code[1, 128, 15] - [1, 128, 40] - [1, 128, 40] - [1, 128, 5*5*138+1]
            self.fc3_w1_size = [self.code3_size[1], 40]
            self.fc3_w2_size = [40, 40]
            self.fc3_w3_size = [40, paramofGen.fc3_filter_size[0] + 1]


            """
            extractor 
            """
            self.extractor_hiddenlayer_size = 2
            # [1, z] - [1, 300] - [1, 300] - [1, prod]
            # extractor fc1 :
            self.extractor_w1 = [z_size_h, 300]
            # extractor fc2 :
            self.extractor_w2 = [300, 300]
            # extractor fc3 :
            self.extractor_w3 = [300, np.prod(self.code1_size) + np.prod(self.code2_size) + np.prod(self.code3_size)]
