import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
import math
from dataloader import dataloader, classifier_dataloader
from model import encoder_cada, decoder_cada, Classifier
from config import opt


class Gzsl_vae():
	"""docstring for Gzsl_vae"""
	def __init__(self, param):
		self.device = torch.device(param.device)

		######################## LOAD DATA #############################
		self.scalar = MinMaxScaler()
		self.trainval_set = dataloader(param, transform=self.scalar, split='trainval')  # AWA2: 23527  40类  CUB:7057 150类
		self.test_set_unseen = dataloader(param, transform=self.scalar, split='test_unseen')  # AWA2: 7913 10类  CUB:2967 50类
		self.test_set_seen = dataloader(param, transform=self.scalar, split='test_seen')  # AWA2: 5882 40类  CUB:1764 150类
		self.trainloader = data.DataLoader(self.trainval_set, batch_size=param.batch_size, shuffle=True)

		self.input_dim = self.trainval_set.__getlen__()
		self.atts_dim = self.trainval_set.__get_attlen__()
		self.num_classes = self.trainval_set.__totalClasses__()
		
		print(30*('-'))
		print("Input_dimension=%d"%self.input_dim)
		print("Attribute_dimension=%d"%self.atts_dim)
		print("z=%d" % param.latent_size)
		print("num_classes=%d"%self.num_classes)
		print(30*('-'))


		####################### INITIALIZE THE MODEL AND OPTIMIZER #####################
		self.model_encoder = encoder_cada(input_dim=self.input_dim,atts_dim=self.atts_dim,z=param.latent_size).to(self.device)
		self.model_decoder = decoder_cada(input_dim=self.input_dim,atts_dim=self.atts_dim,z=param.latent_size).to(self.device)

		learnable_params = list(self.model_encoder.parameters()) + list(self.model_decoder.parameters())
		self.optimizer = optim.Adam(learnable_params, lr=0.00015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

		self.classifier = Classifier(input_dim=param.latent_size,num_class=self.num_classes).to(self.device)
		self.cls_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, betas=(0.5,0.999))

		print(self.model_encoder)
		print(self.model_decoder)
		print(self.classifier)

		################### LOAD PRETRAINED MODEL ########################
		if param.pretrained:
			if param.model_path == '':
				print("Please provide the path of the pretrained model.")
			else:
				checkpoint = torch.load(param.model_path)
				self.model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
				self.model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
				print(">> Pretrained model loaded!")

		########## LOSS ############
		self.l1_loss = nn.L1Loss(reduction='sum')
		self.lossfunction_classifier =  nn.NLLLoss()


		######### Hyper-params #######
		self.gamma = torch.zeros(1, device=self.device).float()
		self.beta = torch.zeros(1, device=self.device).float()
		self.delta = torch.zeros(1, device=self.device).float()



	def train(self, epoch):
		'''
		This function trains the cada_vae model
		'''	
		if epoch > 5 and epoch < 21:
			self.delta += 0.54
		if epoch > 20 and epoch < 76:
			self.gamma += 0.044
		if epoch < 93:
			self.beta += 0.0026

		trainbar = tqdm(self.trainloader)    
		self.model_encoder.train()
		self.model_decoder.train()
		train_loss = 0

		for batch_idx, (x, y, sig) in enumerate(trainbar):
			x.requires_grad = False
			sig.requires_grad = False

			z_img, z_sig, mu_x, logvar_x, mu_sig, logvar_sig = self.model_encoder(x, sig)
			recon_x, recon_sig, sigDecoder_x, xDecoder_sig = self.model_decoder(z_img, z_sig) 
			# loss
			vae_reconstruction_loss = self.l1_loss(recon_x, x) + self.l1_loss(recon_sig, sig)
			cross_reconstruction_loss = self.l1_loss(xDecoder_sig, x) + self.l1_loss(sigDecoder_x, sig)
			KLD_loss = (0.5 * torch.sum(1 + logvar_x - mu_x.pow(2) - logvar_x.exp())) + (0.5 * torch.sum(1 + logvar_sig - mu_sig.pow(2) - logvar_sig.exp()))
			distributed_loss = torch.sqrt(torch.sum((mu_x - mu_sig) ** 2, dim=1) + torch.sum((torch.sqrt(logvar_x.exp()) - torch.sqrt(logvar_sig.exp())) ** 2, dim=1))
			distributed_loss = distributed_loss.sum()

			self.optimizer.zero_grad()

			loss = vae_reconstruction_loss - self.beta*KLD_loss
			if self.delta > 0:
				loss += distributed_loss*self.delta
			if self.gamma > 0:
				loss += cross_reconstruction_loss*self.gamma

			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			trainbar.set_description('l:%.3f' %(train_loss/(batch_idx+1)))
		
		#print("vae %f, da %f, ca %f"%(vae,da,ca))
		print(train_loss/(batch_idx+1))
		
		if epoch%100==0:
			if not os.path.exists('./checkpoints'):
				os.makedirs('./checkpoints')
			name = "./checkpoints/"+"checkpoint_cada_"+opt.dataset_name+".pth"
			torch.save({
				'epoch':epoch,
				'model_encoder_state_dict':self.model_encoder.state_dict(),
				'model_decoder_state_dict':self.model_decoder.state_dict(),				
				'optimizer_state_dict':self.optimizer.state_dict(),
				'loss':loss,
				}, name)
		


	##################### FEATURE EXTRCTION #######################
	def extract_features(self,params):
		print(30*'-')
		print("Preparing dataset for the classifier..")

		self.model_encoder.eval()
		self.model_decoder.eval()

		img_seen_feats = params['img_seen']  # 200
		img_unseen_feats = params['img_unseen']  # 0
		att_seen_feats = params['att_seen']  # 0
		att_unseen_feats = params['att_unseen']  # 400

		seen_classes = self.trainval_set.__NumClasses__()  # 可见类150类
		unseen_classes = self.test_set_unseen.__NumClasses__()  # 不可见类50类

		#atts for unseen classes
		attribute_vector_unseen, labels_unseen = self.test_set_unseen.__attributeVector__()  # shape:50*312  50

		#for trainval features:
		features_seen = []
		labels_seen = []
		k = 0
		for n in seen_classes:  # 150类，每一类200张
			perclass_feats = self.trainval_set.__get_perclass_feats__(n)  # n=1, perclass_feats:45*2048  n=2,perclass_feats:38*2048
			k += perclass_feats.shape[0]
			repeat_factor = math.ceil(img_seen_feats/perclass_feats.shape[0])  # 向上取整 n=1, repeat_factor=5 n=2,repeat_factor=6
			perclass_X = np.repeat(perclass_feats.cpu().numpy(), repeat_factor, axis=0)  # n=1, 225*2048 n=2 228*2048 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!为什么复制重复的
			perclass_labels = torch.from_numpy(np.repeat(n.cpu().numpy(), img_seen_feats, axis=0)).long()  # 200
			seen_feats = perclass_X[:img_seen_feats].astype(np.float32)
			# if seen_feats.shape[0] < 200:
			# 	print(n,"-------", seen_feats.shape)
			features_seen.append(torch.from_numpy(seen_feats))
			labels_seen.append(perclass_labels)
		print("Number of seen features:", k)


		tensor_seen_features = torch.cat(features_seen).to(self.device)  # 30000*2048
		tensor_seen_feats_labels = torch.cat(labels_seen)
		tensor_unseen_attributes = torch.from_numpy(  # 20000*312
			np.repeat(attribute_vector_unseen, att_unseen_feats, axis=0)).float().to(self.device)  # 复制400次：50*400
		tensor_unseen_labels = torch.from_numpy(np.repeat(labels_unseen, att_unseen_feats,axis=0)).long()  # 20000

		test_unseen_X, test_unseen_Y = self.test_set_unseen.__Test_Features_Labels__()  # 2967*2048    2967*1
		test_seen_X, test_seen_Y = self.test_set_seen.__Test_Features_Labels__()  # 1764*2048   1764*1

		with torch.no_grad():
			z_img, z_att, mu_x, logvar_x, mu_att, logvar_att = self.model_encoder(tensor_seen_features, tensor_unseen_attributes)  # 括号内的30000*2048,20000*312！！！！！！！！！！！！！！！！！！！！！！！
			z_unseen_test_img, z_unseen_test_att, mu_x_unseen, logvar_x, mu_att, logvar_att = self.model_encoder(test_unseen_X, tensor_unseen_attributes)  # 括号内的：2967*2048,20000*312
			z_seen_test_img, z_unseen_test_att, mu_x_seen, logvar_x, mu_att, logvar_att = self.model_encoder(test_seen_X, tensor_unseen_attributes)  # 1764*2048， 20000*312

			train_features = torch.cat((z_att, z_img))  # 50000*64
			train_labels = torch.cat((tensor_unseen_labels, tensor_seen_feats_labels))  # 50000

		test_unseen_Y = torch.squeeze(test_unseen_Y)  # 2967
		test_seen_Y = torch.squeeze(test_seen_Y)  # 1764

		print(">> Extraction of trainval, test seen, and test unseen features are complete!")
		print(train_features.shape, train_labels.shape)
		#return train_features, train_labels, z_unseen_test_img, test_unseen_Y, z_seen_test_img, test_seen_Y
		return train_features, train_labels, mu_x_unseen, test_unseen_Y, mu_x_seen, test_seen_Y


	##################### TRAINING THE CLASSIFIER #######################
	def train_classifier(self,epochs):
		train_features, train_labels, test_unseen_features, test_unseen_labels, test_seen_features, test_seen_labels = \
			self.extract_features(params)
		if not os.path.exists('./datas/'+opt.dataset_name+'_features'):
			os.makedirs('./datas/'+opt.dataset_name+'_features')
		np.save('./datas/'+opt.dataset_name+'_features/'+'test_novel_Y.npy', test_unseen_labels.cpu().numpy())

		self.cls_trainData = classifier_dataloader(features_img=train_features, labels=train_labels, device=self.device)  # 50000*64
		self.cls_trainloader = data.DataLoader(self.cls_trainData, batch_size=32, shuffle=True)

		self.cls_test_unseen = classifier_dataloader(features_img=test_unseen_features, labels=test_unseen_labels, device=self.device)  # test_unseen_features为VAE的均值
		self.cls_test_unseenLoader = data.DataLoader(self.cls_test_unseen, batch_size=32, shuffle=False)
		self.test_unseen_target_classes = self.cls_test_unseen.__targetClasses__()		

		self.cls_test_seen = classifier_dataloader(features_img=test_seen_features, labels=test_seen_labels, device=self.device)  # test_seen_features为VAE的均值
		self.cls_test_seenLoader = data.DataLoader(self.cls_test_seen, batch_size=32, shuffle=False)
		self.test_seen_target_classes = self.cls_test_seen.__targetClasses__()

		def val_gzsl(testbar_cls):
			with torch.no_grad():
				self.classifier.eval()
				print("**Validation**")
				preds = []
				target = []
				for batch_idx, (x, y) in enumerate(testbar_cls):
					output = self.classifier(x)
					output_data = torch.argmax(output.data, 1)
					preds.append(output_data)
					target.append(y)
				predictions = torch.cat(preds)
				targets = torch.cat(target)
				return predictions, targets

		best_H = -1
		best_seen = 0
		best_unseen = 0

		############## TRAINING ####################
		for epoch in range(1, epochs+1):  # 1-41
			print("Training: Epoch - ", epoch)
			self.classifier.train()
			trainbar_cls = tqdm(self.cls_trainloader)  # cls_trainloader由50000*64的数据构成
			train_loss = 0
			for batch_idx, (x, y) in enumerate(trainbar_cls):   # y从0开始
				output = self.classifier(x.to(opt.device))
				loss = self.lossfunction_classifier(output, y)
				self.cls_optimizer.zero_grad()
				loss.backward()
				self.cls_optimizer.step()
				train_loss += loss.item()
				trainbar_cls.set_description('l:%.3f' %(train_loss/(batch_idx+1)))

			########## VALIDATION ##################
			accu_unseen = 0
			accu_seen = 0

			testbar_cls_unseen = tqdm(self.cls_test_unseenLoader)  # 由2967*64的数据构成
			testbar_cls_seen = tqdm(self.cls_test_seenLoader)  # 由1764*64的数据构成

			preds_unseen, target_unseen = val_gzsl(testbar_cls_unseen)
			preds_seen, target_seen = val_gzsl(testbar_cls_seen)

			########## ACCURACY METRIC ##################
			def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
				per_class_accuracies = torch.zeros(target_classes.shape[0]).float()
				predicted_label = predicted_label
				for i in range(target_classes.shape[0]):
					is_class = test_label==target_classes[i]  # 找出是该类的
					per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(), is_class.sum().float())
				return per_class_accuracies.mean()

			##################################
			'''
			For NLLL loss the labels are 
			mapped from 0-n, map them back to 1-n 
			for calculating accuracies.
			'''
			target_unseen = target_unseen + 1
			preds_unseen = preds_unseen + 1
			target_seen = target_seen + 1
			preds_seen = preds_seen + 1
			##################################


			accu_unseen = compute_per_class_acc_gzsl(target_unseen, preds_unseen, self.test_unseen_target_classes)
			accu_seen = compute_per_class_acc_gzsl(target_seen, preds_seen, self.test_seen_target_classes)

			if (accu_seen+accu_unseen)>0:
				H = (2*accu_seen*accu_unseen) / (accu_seen+accu_unseen)
			else:
				H = 0

			if H > best_H:

				best_seen = accu_seen
				best_unseen = accu_unseen
				best_H = H

			print(30*'-')
			print('Epoch:', epoch)
			print('Best: u, s, h =%.4f,%.4f,%.4f'%(best_unseen,best_seen,best_H))
			print('u, s, h =%.4f,%.4f,%.4f'%(accu_unseen,accu_seen,H))
			print(30*'-')
				
		return best_seen, best_unseen, best_H



if __name__=='__main__':
	model = Gzsl_vae(opt)
	if not opt.pretrained:
		epochs=100
		for epoch in range(1, epochs + 1):
			print("epoch:", epoch)
			model.train(epoch)

	#CLASSIFIER
	params = {'img_seen':200,
			'img_unseen':0,
			'att_seen':0,
			'att_unseen':400}
	nepochs = 40
	s, u, h = model.train_classifier(nepochs)

	
