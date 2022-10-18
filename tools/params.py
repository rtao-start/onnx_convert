from torchvision.models.efficientnet import MBConvConfig

mbcc1 = MBConvConfig(0.1, 3, 1,1,64,6,5,5)
mbcc2 = MBConvConfig(0.2, 5, 2, 3,32,2,7,7)
mbcc3= MBConvConfig(0.9, 7, 2,5,128,24,13,13)

mbcc = [mbcc1, mbcc2, mbcc3]

param_dict = {}

param_dict['dropout'] = 0.33
param_dict['inverted_residual_setting'] = mbcc

#print('param_dict:', param_dict)
